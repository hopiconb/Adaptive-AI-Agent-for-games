"""
Dual-window FOOTSIES agent visualization.

Runs the trained agent against the built-in CPU bot and shows:
  (1) the Unity FOOTSIES game window at real-time speed
  (2) a pygame schematic view (800×300) of what the neural network sees

Screenshots of the pygame window are saved to ./screenshots/ once per second.

Usage
-----
  python visualize_agent.py                                    # latest model
  python visualize_agent.py --model models/ppo_footsies_final.zip
  python visualize_agent.py --episodes 20
"""
import os
import sys
import glob
import time
import argparse

import numpy as np
import pygame
import gymnasium as gym
import footsies_gym  # noqa: F401 — registers FootsiesEnv-v0

from stable_baselines3 import PPO
from train import NumpyObsWrapper, GAME_PATH
from footsies_gym.moves import FootsiesMove

# ── layout constants ───────────────────────────────────────────────────────────
W, H       = 800, 300
LM, RM     = 65, 65                   # left/right margin
STAGE_W    = W - LM - RM              # 670 px of usable stage width
SX_MIN     = -4.6
SX_MAX     = 4.6
CHAR_Y     = 130                      # y-centre of character circles
WALL_TOP   = CHAR_Y - 58
WALL_BOTTOM = CHAR_Y + 58

# ── colours ────────────────────────────────────────────────────────────────────
BG          = (22,  22,  33)
AXIS_C      = (100, 100, 120)
WALL_C      = (190, 190, 210)
AGENT_C     = (55,  200, 80)
OPP_C       = (215, 60,  60)
WHITE       = (238, 238, 240)
GRAY        = (88,  88,  98)
BTN_ON      = (55,  200, 80)
BTN_OFF     = (65,  65,  78)
OUTLINE     = (160, 160, 175)

# ── game constants ─────────────────────────────────────────────────────────────
MAX_STEPS   = 12_000
MOVE_LIST   = list(FootsiesMove)                  # index 0-16; obs uses 0-14
ACTION_SYMS = ["◀", "▶", "⚔"]                    # left, right, attack


# ── helpers ────────────────────────────────────────────────────────────────────

def sx_to_px(x: float) -> int:
    """Map a stage X coordinate to a horizontal pixel position in the pygame window.

    The FOOTSIES stage spans [SX_MIN, SX_MAX] = [−4.6, 4.6]. The usable pygame canvas
    is [LM, W−RM] = [65, 735] px. Linearly interpolates between these ranges.

    Args:
        x: Stage X coordinate in game units.
    Returns:
        Integer pixel X position within the pygame surface.
    """
    t = (x - SX_MIN) / (SX_MAX - SX_MIN)
    return int(LM + t * STAGE_W)


def move_radius(move: FootsiesMove, frame: int) -> int:
    """Choose the character circle radius based on whether the move's hitbox is live.

    Inflating the circle during active frames gives an immediate visual cue that the
    character is threatening and helps identify when the opponent is punishable.

    Args:
        move: FootsiesMove enum value for the character's current animation.
        frame: Current frame index within that animation.
    Returns:
        28 if the hitbox is active at this frame, 20 otherwise.
    """
    if move.value.active > 0 and move.in_active(frame):
        return 28
    return 20


def draw_frame(screen, font, sfont, obs, action, cum_rew, frame_n, ep, wins):
    """Render one visualization frame to the pygame surface and flip the display.

    Draws the stage axis and walls, two character circles (green = agent, red = CPU)
    scaled by hitbox activity, guard-health bars below each circle, an action button
    bar showing which of the three inputs are active, and a HUD line in the top-right
    corner with episode, win count, cumulative reward, and frame number.

    Args:
        screen: pygame.Surface to draw on.
        font: pygame.Font for action symbols (larger size).
        sfont: pygame.Font for labels and HUD text (smaller size).
        obs: Numpy obs dict from NumpyObsWrapper (position, move, move_frame, guard).
        action: Array-like of 3 ints (the agent's last chosen action).
        cum_rew: Cumulative episode reward up to this frame.
        frame_n: Current step number within the episode.
        ep: Current episode number (1-indexed).
        wins: Total wins accumulated across all episodes so far.
    """
    screen.fill(BG)

    # ── horizontal axis + stage walls ─────────────────────────────────────────
    pygame.draw.line(screen, AXIS_C, (LM, CHAR_Y), (W - RM, CHAR_Y), 1)
    for wx in [SX_MIN, SX_MAX]:
        px = sx_to_px(wx)
        pygame.draw.line(screen, WALL_C, (px, WALL_TOP), (px, WALL_BOTTOM), 2)

    # ── characters (agent = index 0, opponent = index 1) ──────────────────────
    positions   = obs["position"]    # float32 (2,)
    move_idxs   = obs["move"]        # int64   (2,)
    move_frames = obs["move_frame"]  # float32 (2,)
    guards      = obs["guard"]       # int64   (2,)

    char_meta = [(AGENT_C, "AGENT"), (OPP_C, "CPU")]
    for i, (color, label) in enumerate(char_meta):
        cx     = sx_to_px(float(positions[i]))
        move   = MOVE_LIST[int(move_idxs[i])]
        mframe = int(move_frames[i])
        r      = move_radius(move, mframe)

        # circle + outline
        pygame.draw.circle(screen, color,   (cx, CHAR_Y), r)
        pygame.draw.circle(screen, OUTLINE, (cx, CHAR_Y), r, 1)

        # small role label above everything (AGENT / CPU)
        lbl_s = sfont.render(label, True, GRAY)
        screen.blit(lbl_s, (cx - lbl_s.get_width() // 2, CHAR_Y - r - 38))

        # move name + frame counter
        move_txt = f"{move.name} ({mframe})"
        mv_s = sfont.render(move_txt, True, color)
        screen.blit(mv_s, (cx - mv_s.get_width() // 2, CHAR_Y - r - 22))

        # guard bars: 3 small filled rects below the circle
        bw, bh, bgap = 14, 7, 4
        total_bw = 3 * bw + 2 * bgap
        bx0 = cx - total_bw // 2
        by  = CHAR_Y + r + 7
        for g in range(3):
            bx = bx0 + g * (bw + bgap)
            bc = color if g < int(guards[i]) else GRAY
            pygame.draw.rect(screen, bc,    (bx, by, bw, bh))
            pygame.draw.rect(screen, WHITE, (bx, by, bw, bh), 1)

    # ── action button bar (bottom centre) ─────────────────────────────────────
    btn_y  = H - 58
    btn_w, btn_h = 72, 30
    gap    = 14
    total  = 3 * btn_w + 2 * gap
    bx0    = W // 2 - total // 2

    for i, sym in enumerate(ACTION_SYMS):
        bx  = bx0 + i * (btn_w + gap)
        lit = bool(int(action[i]))
        bc  = BTN_ON if lit else BTN_OFF
        pygame.draw.rect(screen, bc,    (bx, btn_y, btn_w, btn_h), border_radius=6)
        pygame.draw.rect(screen, WHITE, (bx, btn_y, btn_w, btn_h), 1, border_radius=6)
        sym_s = font.render(sym, True, WHITE)
        screen.blit(sym_s,
                    (bx + btn_w // 2 - sym_s.get_width() // 2,
                     btn_y + btn_h // 2 - sym_s.get_height() // 2))

    # ── HUD: top-right corner ──────────────────────────────────────────────────
    hud = (f"Ep {ep}  Wins {wins}  "
           f"Reward {cum_rew:+.3f}  Frame {frame_n}")
    hud_s = sfont.render(hud, True, WHITE)
    screen.blit(hud_s, (W - hud_s.get_width() - 8, 6))

    pygame.display.flip()


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    """Run the dual-window visualization: Unity game + pygame schematic overlay.

    Loads a PPO model and opens two windows: the Unity FOOTSIES game at real-time
    speed and a pygame 800×300 schematic showing positions, active hitboxes, guard
    bars, and the agent's action each frame. Saves a screenshot of the pygame window
    once per wall-clock second to ./screenshots/. ESC or closing the pygame window
    terminates the loop cleanly via SystemExit, which triggers the finally block.
    """
    parser = argparse.ArgumentParser(
        description="Watch a trained FOOTSIES agent with a pygame neural-network view")
    parser.add_argument("--model", default=None,
                        help="Path to .zip checkpoint. Default: latest in models/")
    parser.add_argument("--episodes", type=int, default=10,
                        help="Number of episodes to run")
    args = parser.parse_args()

    # locate model
    model_path = args.model
    if model_path is None:
        zips = sorted(glob.glob(os.path.join("models", "*.zip")))
        if not zips:
            raise FileNotFoundError("No .zip checkpoints found in models/")
        model_path = zips[-1]
    print(f"Model: {model_path}")
    model = PPO.load(model_path)

    # build env  — real-time, Unity window visible, dense reward for richer HUD
    env = gym.make(
        "FootsiesEnv-v0",
        game_path=GAME_PATH,
        render_mode="human",
        fast_forward=False,
        sync_mode="synced_non_blocking",
        dense_reward=True,
        disable_env_checker=True,
    )
    env = NumpyObsWrapper(env)

    # pygame init
    pygame.init()
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("FOOTSIES — Agent Neural Network View")

    # prefer a font with good Unicode coverage (for ◀ ▶ ⚔ glyphs)
    _fn = pygame.font.match_font(
        "dejavusans,dejavusansmono,freesans,ubuntumono,unifont,arial,sansserif"
    )
    font  = pygame.font.Font(_fn, 22) if _fn else pygame.font.SysFont(None, 24)
    sfont = pygame.font.Font(_fn, 15) if _fn else pygame.font.SysFont(None, 17)

    # screenshot directory
    ss_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "screenshots")
    os.makedirs(ss_dir, exist_ok=True)
    last_ss = time.time()

    wins = 0
    print(f"Watching {args.episodes} episodes…  (ESC or close window to quit)\n")

    try:
        for ep in range(1, args.episodes + 1):
            obs, _   = env.reset()
            terminated = truncated = False
            steps      = 0
            cum_rew    = 0.0
            action     = np.zeros(3, dtype=np.int64)  # shown before first step

            while not (terminated or truncated) and steps < MAX_STEPS:
                # keep pygame event queue drained so the window stays responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        raise SystemExit
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                        raise SystemExit

                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                cum_rew += float(reward)
                steps   += 1

                draw_frame(screen, font, sfont, obs, action,
                           cum_rew, steps, ep, wins)

                # screenshot once per wall-clock second
                now = time.time()
                if now - last_ss >= 1.0:
                    pygame.image.save(
                        screen,
                        os.path.join(ss_dir, f"frame_{int(now)}.png"),
                    )
                    last_ss = now

                time.sleep(1 / 60)

            won = bool(terminated and cum_rew > 0)
            wins += won
            result = "WIN " if won else "LOSS"
            print(f"  Episode {ep:3d}: {result}  steps={steps:4d}  "
                  f"reward={cum_rew:+.3f}  win rate={wins/ep:.1%}")

    except SystemExit:
        print("\nWindow closed.")
    except KeyboardInterrupt:
        print("\nInterrupted.")
    finally:
        pygame.quit()
        env.close()


if __name__ == "__main__":
    main()
