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
    """Converts the env's tuple obs values to numpy arrays — required for SB3."""
    def observation(self, obs):
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
        super().__init__(env)
        self.gamma = gamma
        self._phi = 0.0

    def _potential(self, obs: dict) -> float:
        return float(obs["guard"][0] - obs["guard"][1]) / 3.0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._phi = self._potential(obs)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        phi_next = self._potential(obs)
        shaping = self.gamma * phi_next - self._phi
        self._phi = phi_next if not (terminated or truncated) else 0.0
        return obs, reward + shaping, terminated, truncated, info


def make_env(fast_forward_speed: float = 20.0):
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
        env = PBRSWrapper(env, gamma=0.99)
        return env
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train PPO on FOOTSIES")
    parser.add_argument("--steps", type=int, default=500_000,
                        help="Total training timesteps")
    parser.add_argument("--fast-forward-speed", type=float, default=20.0,
                        help="Game speed multiplier (base: 50 updates/s)")
    parser.add_argument("--save-freq", type=int, default=50_000,
                        help="Checkpoint every N steps")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to .zip checkpoint to resume from")
    args = parser.parse_args()

    os.makedirs(LOGS_DIR, exist_ok=True)
    os.makedirs(MODELS_DIR, exist_ok=True)

    vec_env = DummyVecEnv([make_env(args.fast_forward_speed)])
    vec_env = VecMonitor(vec_env, filename=os.path.join(LOGS_DIR, "monitor"))

    if args.resume:
        print(f"Resuming from {args.resume}")
        model = PPO.load(args.resume, env=vec_env, tensorboard_log=LOGS_DIR)
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
            tensorboard_log=LOGS_DIR,
            verbose=1,
        )

    checkpoint_cb = CheckpointCallback(
        save_freq=args.save_freq,
        save_path=MODELS_DIR,
        name_prefix="ppo_footsies",
        verbose=1,
    )

    print(f"Training for {args.steps:,} steps | speed: {args.fast_forward_speed}x | "
          f"checkpoints every {args.save_freq:,} steps")
    print(f"TensorBoard: tensorboard --logdir {LOGS_DIR}")

    model.learn(
        total_timesteps=args.steps,
        callback=checkpoint_cb,
        progress_bar=True,
        reset_num_timesteps=args.resume is None,
        tb_log_name="PPO_footsies",
    )

    final_path = os.path.join(MODELS_DIR, "ppo_footsies_final")
    model.save(final_path)
    print(f"\nDone. Final model: {final_path}.zip")
    vec_env.close()


if __name__ == "__main__":
    main()
