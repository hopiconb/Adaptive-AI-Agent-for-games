"""
PPO training for FOOTSIES using Stable-Baselines3.
Runs headless (-batchmode -nographics), fast-forward enabled.

Usage:
  python train.py                        # 500k steps, 20x speed
  python train.py --steps 1000000       # 1M steps
  python train.py --fast-forward-speed 10
"""
import os
import argparse
import numpy as np
import gymnasium as gym
import footsies_gym  # registers FootsiesEnv-v0

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback

GAME_PATH = os.path.join(os.path.dirname(__file__), "footsies-env", "Build", "FOOTSIES")
LOGS_DIR  = os.path.join(os.path.dirname(__file__), "logs")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


class NumpyObsWrapper(gym.ObservationWrapper):
    """Observation wrapper that sits at the bottom of every wrapper stack in this project.
    FootsiesEnv returns Python tuples for every obs field; Stable-Baselines3 requires
    numpy arrays. This wrapper converts each field in-place and fixes dtypes so
    MultiInputPolicy receives correctly typed inputs without further casting."""

    def observation(self, obs):
        """Convert each obs field from a Python tuple to a typed numpy array.

        Args:
            obs: Raw obs dict from FootsiesEnv with tuple values.
        Returns:
            Dict with the same keys and numpy-array values of the expected dtypes.
        """
        return {
            "guard":      np.array(obs["guard"],      dtype=np.int64),
            "move":       np.array(obs["move"],       dtype=np.int64),
            "move_frame": np.array(obs["move_frame"], dtype=np.float32),
            "position":   np.array(obs["position"],  dtype=np.float32),
        }


class PBRSWrapper(gym.Wrapper):
    """
    Potential-Based Reward Shaping (Ng, Harada, Russell 1999).

    Φ(s) = (p1_guard - p2_guard) / 3   [range: -1 .. +1]
    F(s→s') = γ·Φ(s') - Φ(s)

    This strictly preserves the optimal policy while giving the agent
    denser signal about guard-health advantage between steps.
    Gamma must match PPO's discount to avoid shifting value estimates.
    """
    def __init__(self, env: gym.Env, gamma: float = 0.99):
        """Initialise the wrapper and cache the discount factor used for shaping.

        Args:
            env: The environment to wrap (should already have NumpyObsWrapper applied).
            gamma: Discount factor; must match the PPO agent's gamma so that the
                   shaping term does not alter the value function scale.
        """
        super().__init__(env)
        self.gamma = gamma
        self._phi = 0.0

    def _potential(self, obs: dict) -> float:
        """Compute Φ(s) = (p1_guard − p2_guard) / 3, normalised to [−1, +1].

        Args:
            obs: Numpy obs dict as produced by NumpyObsWrapper.
        Returns:
            Scalar float representing the guard-advantage potential at state s.
        """
        return float(obs["guard"][0] - obs["guard"][1]) / 3.0

    def reset(self, **kwargs):
        """Reset the environment and seed the initial potential Φ(s₀).

        Returns:
            (obs, info) from the underlying environment.
        """
        obs, info = self.env.reset(**kwargs)
        self._phi = self._potential(obs)
        return obs, info

    def step(self, action):
        """Step the environment and augment the reward with the PBRS shaping term.

        The shaped reward is r' = r + γ·Φ(s') − Φ(s). The potential is zeroed at
        episode boundaries so no shaping leaks across episodes.

        Args:
            action: Action chosen by the agent.
        Returns:
            (obs, shaped_reward, terminated, truncated, info) tuple.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = self._potential(obs)
        shaping = self.gamma * phi_next - self._phi
        self._phi = phi_next if not (terminated or truncated) else 0.0
        return obs, reward + shaping, terminated, truncated, info


def make_env(fast_forward_speed: float = 20.0, use_pbrs: bool = True):
    """Return a zero-argument factory function that builds one wrapped FootsiesEnv.

    The factory pattern is required by DummyVecEnv / SubprocVecEnv, which call the
    returned callable once per parallel worker. NumpyObsWrapper is always applied;
    PBRSWrapper is applied only when use_pbrs is True.

    Args:
        fast_forward_speed: Game speed multiplier passed to FootsiesEnv.
        use_pbrs: If True, wraps the env in PBRSWrapper(gamma=0.99).
    Returns:
        A zero-argument callable that constructs and returns the wrapped env.
    """
    def _init():
        env = gym.make(
            "FootsiesEnv-v0",
            game_path=GAME_PATH,
            render_mode=None,          # headless: -batchmode -nographics
            fast_forward=True,
            fast_forward_speed=fast_forward_speed,
            sync_mode="synced_non_blocking",
            disable_env_checker=True,  # suppress tuple-vs-ndarray warnings
        )
        env = NumpyObsWrapper(env)
        if use_pbrs:
            env = PBRSWrapper(env, gamma=0.99)
        return env
    return _init


def main():
    """Parse CLI arguments, build the vectorised environment, instantiate or resume a PPO
    model, attach a CheckpointCallback, run model.learn(), and save the final model.

    Directories (logs and checkpoints) are chosen automatically based on --no-pbrs:
    plain runs write to logs/ and models/; ablation runs write to logs_no_pbrs/ and
    models_no_pbrs/ so the two experiments never collide.
    """
    parser = argparse.ArgumentParser(description="Train PPO on FOOTSIES")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--fast-forward-speed", type=float, default=20.0,
                        help="Game speed multiplier (base: 50 updates/s)")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .zip checkpoint to resume from")
    parser.add_argument("--no-pbrs", action="store_true",
                        help="Disable PBRS shaping; train on pure sparse terminal reward")
    args = parser.parse_args()

    base = os.path.dirname(__file__)
    logs_dir   = os.path.join(base, "logs_no_pbrs"   if args.no_pbrs else "logs")
    models_dir = os.path.join(base, "models_no_pbrs" if args.no_pbrs else "models")

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    use_pbrs = not args.no_pbrs
    vec_env = DummyVecEnv([make_env(args.fast_forward_speed, use_pbrs=use_pbrs)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(logs_dir, "monitor"))

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=logs_dir)
    else:
        model = PPO(
            policy="MultiInputPolicy",
            env=vec_env,
            # Core PPO hyperparameters
            learning_rate=3e-4,
            n_steps=2048,          # steps per rollout per env
            batch_size=64,
            n_epochs=10,
            gamma=0.99,            # must match PBRSWrapper.gamma
            gae_lambda=0.95,
            clip_range=0.2,
            # Entropy bonus: important for fighting games (encourages exploration)
            ent_coef=0.01,
            # Value loss coefficient
            vf_coef=0.5,
            max_grad_norm=0.5,
            tensorboard_log=logs_dir,
            verbose=1,
        )

    reward_mode = "sparse (no PBRS)" if args.no_pbrs else "PBRS-shaped"
    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=models_dir,
        name_prefix="ppo_footsies",
        verbose=1,
    )

    print(f"Training for {args.steps:,} steps | speed: {args.fast_forward_speed}x | "
          f"reward: {reward_mode} | checkpoints every {args.save_freq:,} steps")
    print(f"TensorBoard: tensorboard --logdir {logs_dir}")

    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_cb,
        progress_bar=True,
        reset_num_timesteps=args.resume is None,
        tb_log_name="PPO_footsies",
    )

    final_path = os.path.join(models_dir, "ppo_footsies_final")
    model.save(final_path)
    print(f"\nDone. Final model: {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
