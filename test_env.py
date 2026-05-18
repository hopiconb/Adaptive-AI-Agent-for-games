"""
Sanity test for FootsiesEnv.
Requires the game binary at GAME_PATH (build it from footsies-env/ with Unity 2022.3.10f1).
Run headless (no display needed).
"""
import os
import sys

GAME_PATH = os.path.join(os.path.dirname(__file__), "footsies-env", "Build", "FOOTSIES")

import gymnasium
import footsies_gym  # registers FootsiesEnv-v0

def main():
    print(f"[footsies-env] game binary: {GAME_PATH}")
    if not os.path.exists(GAME_PATH):
        print(f"ERROR: game binary not found at {GAME_PATH}")
        print("Build the Unity project first (see README / Unity setup instructions).")
        sys.exit(1)

    env = gymnasium.make(
        "FootsiesEnv-v0",
        game_path=GAME_PATH,
        render_mode=None,       # headless: -batchmode -nographics
        fast_forward=True,
        fast_forward_speed=6.0,
        sync_mode="synced_non_blocking",
    )

    print(f"\nobservation_space:\n  {env.observation_space}")
    print(f"\naction_space:\n  {env.action_space}")
    print(f"\nsample observation (from space, no game needed):\n  {env.observation_space.sample()}")

    print("\nResetting environment (launches game process)...")
    obs, info = env.reset(seed=42)
    print(f"Initial obs: {obs}")

    total_reward = 0.0
    steps = 0
    terminated = truncated = False

    print("\nRunning 100 steps with random actions...")
    while steps < 100:
        if terminated or truncated:
            obs, info = env.reset()
            terminated = truncated = False

        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1

    env.close()
    print(f"\nDone. Steps: {steps} | Total reward: {total_reward:.4f}")

if __name__ == "__main__":
    main()
