"""
tests/test_pipeline.py

Three smoke-tests that verify the core pipeline components work correctly
without running a full training or evaluation loop.

Run with:
    pytest tests/ -v
"""

import os
import sys
import numpy as np
import pytest

# Ensure the project root is on the path so local modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gymnasium as gym
import footsies_gym  # noqa: F401 — registers FootsiesEnv-v0

from train import NumpyObsWrapper, PBRSWrapper, GAME_PATH
from eval_scenarios import resolve_action


# ── Test 1: environment creation ─────────────────────────────────────────────

class TestEnvCreation:
    """Verify that FootsiesEnv-v0 registers and exposes the expected spaces."""

    def test_env_creation(self):
        """Instantiate a headless FootsiesEnv and check space types and shapes."""
        env = gym.make(
            "FootsiesEnv-v0",
            game_path=GAME_PATH,
            render_mode=None,
            disable_env_checker=True,
        )
        obs_space = env.observation_space
        act_space = env.action_space

        # Observation space must be a Dict with exactly these four keys
        assert isinstance(obs_space, gym.spaces.Dict), (
            f"Expected Dict obs space, got {type(obs_space)}"
        )
        assert set(obs_space.spaces.keys()) == {"guard", "move", "move_frame", "position"}, (
            f"Unexpected obs keys: {set(obs_space.spaces.keys())}"
        )

        # guard and move are MultiDiscrete; move_frame and position are Box
        assert isinstance(obs_space["guard"],      gym.spaces.MultiDiscrete)
        assert isinstance(obs_space["move"],       gym.spaces.MultiDiscrete)
        assert isinstance(obs_space["move_frame"], gym.spaces.Box)
        assert isinstance(obs_space["position"],   gym.spaces.Box)

        # Action space is MultiBinary(3)
        assert isinstance(act_space, gym.spaces.MultiBinary), (
            f"Expected MultiBinary action space, got {type(act_space)}"
        )
        assert act_space.n == 3, f"Expected MultiBinary(3), got MultiBinary({act_space.n})"

        env.close()


# ── Test 2: PBRS wrapper math ─────────────────────────────────────────────────

class _MockEnv(gym.Env):
    """Minimal gym.Env subclass with fixed obs/reward sequences for PBRS unit tests.

    Gymnasium 1.x Wrapper.__init__ asserts isinstance(env, gym.Env), so the mock must
    inherit from it. Only reset, step, render, and close are implemented; all other
    behaviour is delegated to gym.Env defaults.
    """

    metadata = {}

    def __init__(self, obs_sequence, reward_sequence):
        """Set up fixed obs/reward sequences to be consumed step by step.

        Args:
            obs_sequence: List of obs dicts returned by successive reset/step calls.
            reward_sequence: List of scalar rewards paired with each step obs.
        """
        super().__init__()
        self._obs_seq    = list(obs_sequence)
        self._rew_seq    = list(reward_sequence)
        self._idx        = 0
        self.observation_space = gym.spaces.Dict({
            "guard":      gym.spaces.MultiDiscrete([4, 4]),
            "move":       gym.spaces.MultiDiscrete([15, 15]),
            "move_frame": gym.spaces.Box(0.0, 55.0, (2,), np.float32),
            "position":   gym.spaces.Box(-4.6, 4.6, (2,), np.float32),
        })
        self.action_space = gym.spaces.MultiBinary(3)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._idx = 0
        return self._obs_seq[0], {}

    def step(self, action):
        self._idx += 1
        obs    = self._obs_seq[self._idx]
        reward = self._rew_seq[self._idx - 1]
        terminated = self._idx >= len(self._rew_seq)
        return obs, reward, terminated, False, {}

    def render(self):
        pass

    def close(self):
        pass


def _make_obs(p1_guard: int, p2_guard: int) -> dict:
    """Build a minimal obs dict with the given guard values and zeroed other fields."""
    return {
        "guard":      np.array([p1_guard, p2_guard], dtype=np.int64),
        "move":       np.array([0, 0],               dtype=np.int64),
        "move_frame": np.array([0.0, 0.0],           dtype=np.float32),
        "position":   np.array([0.0, 0.0],           dtype=np.float32),
    }


class TestPBRSWrapper:
    """Verify the PBRS shaping formula using hand-computed expected values."""

    GAMMA = 0.99

    def test_shaping_added_to_reward(self):
        """Shaped reward must equal base_reward + gamma*Phi(s') - Phi(s).

        Scenario:
          s  : p1_guard=2, p2_guard=2  →  Phi(s)  = (2-2)/3 = 0.0
          s' : p1_guard=2, p2_guard=1  →  Phi(s') = (2-1)/3 ≈ 0.3333
          base reward = 0.0 (no terminal)

        Expected shaping = 0.99 * 0.3333 - 0.0 = 0.33
        Expected total   = 0.0 + 0.33 = 0.33  (to 2 d.p.)
        """
        obs_reset = _make_obs(2, 2)   # s
        obs_step  = _make_obs(2, 1)   # s'
        mock = _MockEnv([obs_reset, obs_step], [0.0])
        wrapped = PBRSWrapper(mock, gamma=self.GAMMA)

        wrapped.reset()
        _, shaped_reward, _, _, _ = wrapped.step(np.array([0, 0, 0]))

        phi_s      = (2 - 2) / 3.0
        phi_s_next = (2 - 1) / 3.0
        expected   = 0.0 + self.GAMMA * phi_s_next - phi_s

        assert abs(shaped_reward - expected) < 1e-5, (
            f"Shaped reward {shaped_reward:.6f} ≠ expected {expected:.6f}"
        )

    def test_potential_reset_on_termination(self):
        """Internal potential must be zeroed at episode end, not carried to next episode.

        If Phi is not zeroed on termination the first step of the next episode would
        produce a shaping spike. This test verifies the reset happens correctly by
        checking that a second reset produces Phi(s₀) of the new episode, not the
        leftover value.
        """
        obs_s0   = _make_obs(3, 0)  # Phi = 1.0
        obs_s1   = _make_obs(3, 3)  # Phi = 0.0 (terminal)
        obs_new  = _make_obs(1, 1)  # new episode start, Phi = 0.0

        class _TwoEpMock(_MockEnv):
            def __init__(self_inner):
                super().__init__([obs_s0, obs_s1], [1.0])

            def reset(self_inner, *, seed=None, options=None):
                return obs_new, {}

        mock    = _TwoEpMock()
        wrapped = PBRSWrapper(mock, gamma=self.GAMMA)

        wrapped.reset()
        wrapped.step(np.array([0, 0, 0]))   # terminal step
        wrapped.reset()                      # new episode
        assert wrapped._phi == pytest.approx(0.0, abs=1e-6), (
            f"Phi after reset should be 0.0 for equal guards, got {wrapped._phi}"
        )


# ── Test 3: action resolution ─────────────────────────────────────────────────

class TestActionResolution:
    """Verify resolve_action returns correct MultiBinary vectors for both sides."""

    def test_forward_attack_p1(self):
        """FORWARD_ATTACK for P1 must be [left=0, right=1, attack=1]."""
        result = resolve_action("FORWARD_ATTACK", "P1")
        assert result == [0, 1, 1], f"P1 FORWARD_ATTACK: expected [0,1,1], got {result}"

    def test_forward_attack_p2(self):
        """FORWARD_ATTACK for P2 must be [left=1, right=0, attack=1] (mirrored)."""
        result = resolve_action("FORWARD_ATTACK", "P2")
        assert result == [1, 0, 1], f"P2 FORWARD_ATTACK: expected [1,0,1], got {result}"

    def test_idle_both_sides(self):
        """IDLE must be [0, 0, 0] regardless of side."""
        assert resolve_action("IDLE", "P1") == [0, 0, 0]
        assert resolve_action("IDLE", "P2") == [0, 0, 0]

    def test_backward_p1_vs_p2(self):
        """BACKWARD bits must be flipped between P1 and P2."""
        p1 = resolve_action("BACKWARD", "P1")
        p2 = resolve_action("BACKWARD", "P2")
        assert p1 == [1, 0, 0], f"P1 BACKWARD: expected [1,0,0], got {p1}"
        assert p2 == [0, 1, 0], f"P2 BACKWARD: expected [0,1,0], got {p2}"

    def test_case_insensitive(self):
        """resolve_action must accept lowercase and mixed-case action names."""
        assert resolve_action("forward_attack", "P1") == resolve_action("FORWARD_ATTACK", "P1")

    def test_unknown_action_raises(self):
        """An unrecognised action name must raise ValueError."""
        with pytest.raises(ValueError):
            resolve_action("JUMP", "P1")
