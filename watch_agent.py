"""
Live render of a trained FOOTSIES agent (requires a display).

Loads a PPO model and runs it against the built-in CPU bot with render_mode="human"
so the Unity game window is shown. Prints per-episode win/loss and running win rate.

Usage
-----
  python watch_agent.py                                    # latest model
  python watch_agent.py --model models/ppo_footsies_final.zip
  python watch_agent.py --episodes 20 --speed 1.0         # real-time speed
"""
import os
import argparse
import glob
import gymnasium as gym
import footsies_gym  # noqa: F401 — registers FootsiesEnv-v0

from stable_baselines3 import PPO
from train import NumpyObsWrapper, GAME_PATH

MAX_STEPS = 12_000


def main():
    """Load a PPO model and run it visually against the built-in CPU bot.

    Opens the Unity game window in render_mode="human" (requires a display). Prints
    per-episode win/loss, step count, and running win rate to stdout. Catches
    KeyboardInterrupt so the env is always closed cleanly. A step cap of MAX_STEPS
    per episode prevents hangs if neither player can finish.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to .zip checkpoint. Default: latest in models/")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to watch")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Game speed multiplier (1.0 = real-time)")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        zips = sorted(glob.glob(os.path.join("models", "*.zip")))
        if not zips:
            raise FileNotFoundError("No .zip checkpoints in models/")
        model_path = zips[-1]
    print(f"Model: {model_path}")

    model = PPO.load(model_path)

    env = gym.make(
        "FootsiesEnv-v0",
        game_path=GAME_PATH,
        render_mode="human",
        fast_forward=args.speed > 1.0,
        fast_forward_speed=args.speed,
        sync_mode="synced_non_blocking",
        dense_reward=False,
        disable_env_checker=True,
    )
    env = NumpyObsWrapper(env)

    wins = 0
    print(f"\nWatching {args.episodes} episodes at {args.speed}x speed…\n")

    try:
        for ep in range(1, args.episodes + 1):
            obs, _ = env.reset()
            terminated = truncated = False
            steps = 0
            reward = 0.0
            while not (terminated or truncated) and steps < MAX_STEPS:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                steps += 1

            won = bool(terminated and reward > 0)
            wins += won
            result = "WIN " if won else "LOSS"
            print(f"  Episode {ep:3d}: {result}  "
                  f"steps={steps:4d}  "
                  f"win rate={wins/ep:.1%}")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        env.close()


if __name__ == "__main__":
    main()