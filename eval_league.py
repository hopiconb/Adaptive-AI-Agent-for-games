"""
League evaluation for a trained FOOTSIES PPO agent.

Runs 100 matches against three baselines:
  1. random   – uniform random MultiBinary(3) actions
  2. backward – always moves away from the opponent
  3. cpu_bot  – the game's built-in AI (always agent=P1; see NOTE below)

For random and backward, 50 matches are played with the agent as P1 and 50 with the
agent as P2 (roles swapped via the FootsiesEnv opponent socket). For cpu_bot the
game's built-in bot must occupy the P2 slot, so all 100 matches are agent=P1.

Outputs
-------
  results/league_results.csv  – one row per match
  results/league_summary.json – win rates, Wilson 95% CI, Wilcoxon p (Bonferroni-corrected)

Usage
-----
  python eval_league.py                          # uses latest checkpoint
  python eval_league.py --model models/ppo_footsies_500000_steps.zip
  python eval_league.py --speed 20 --matches 100
"""
import os
import csv
import json
import math
import time
import glob
import argparse
import numpy as np
import gymnasium as gym
import footsies_gym  # registers FootsiesEnv-v0

from scipy.stats import wilcoxon as scipy_wilcoxon
from stable_baselines3 import PPO
from train import NumpyObsWrapper, GAME_PATH

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
BONFERRONI_N = 3          # number of simultaneous tests
ALPHA_BASE   = 0.05
ALPHA        = ALPHA_BASE / BONFERRONI_N   # 0.01667
MAX_STEPS    = 12_000     # step cap per match (~4 min of real-time game)


# ── Observation utilities ─────────────────────────────────────────────────────

def to_numpy_obs(obs: dict) -> dict:
    """Convert a raw FootsiesEnv observation (Python tuples) to typed numpy arrays.

    Used when the environment was not wrapped with NumpyObsWrapper, e.g., inside
    opponent callbacks that receive the raw obs directly from the engine.

    Args:
        obs: Dict with tuple values as returned by FootsiesEnv.
    Returns:
        Dict with the same keys and correctly typed numpy array values.
    """
    return {
        "guard":      np.array(obs["guard"],      dtype=np.int64),
        "move":       np.array(obs["move"],       dtype=np.int64),
        "move_frame": np.array(obs["move_frame"], dtype=np.float32),
        "position":   np.array(obs["position"],  dtype=np.float32),
    }


def swap_to_p2_perspective(obs: dict) -> dict:
    """Reframe a P1-perspective observation as if the P2 player were P1.

    FootsiesEnv always emits observations from P1's point of view. When the trained
    agent plays as P2, its learned policy expects to be at index 0; this function
    swaps the two player slots and negates positions so the coordinate origin is
    mirrored, preserving the spatial semantics of the policy.

    Args:
        obs: Numpy or tuple obs dict in P1-perspective format.
    Returns:
        New obs dict where guard/move/move_frame slots are exchanged and positions
        are negated, giving the agent a consistent P1-like view.
    """
    return {
        "guard":      np.array([obs["guard"][1],      obs["guard"][0]],      dtype=np.int64),
        "move":       np.array([obs["move"][1],        obs["move"][0]],        dtype=np.int64),
        "move_frame": np.array([obs["move_frame"][1],  obs["move_frame"][0]],  dtype=np.float32),
        "position":   np.array([-obs["position"][1],   -obs["position"][0]],   dtype=np.float32),
    }


# ── Baseline policies ─────────────────────────────────────────────────────────

def random_policy(obs, info=None):
    """Baseline policy that samples a uniformly random MultiBinary(3) action each step.

    Args:
        obs: Ignored — present only to satisfy the opponent-callable signature.
        info: Ignored.
    Returns:
        A 3-tuple of booleans drawn independently from Bernoulli(0.5).
    """
    return tuple(bool(x) for x in np.random.randint(0, 2, 3))


def backward_policy_as_p2(obs, info=None):
    """Baseline policy for the P2 slot: always move away from the opponent.

    Reads positions from the raw P1-perspective obs (index 0 = P1, index 1 = P2) and
    returns the MultiBinary action that moves P2 further from P1.

    Args:
        obs: Raw or numpy obs dict in P1-perspective format.
        info: Ignored.
    Returns:
        3-tuple (left, right, attack) booleans directing P2 away from P1.
    """
    p1_pos, p2_pos = float(obs["position"][0]), float(obs["position"][1])
    if p2_pos >= p1_pos:
        return (False, True, False)   # move right (further from P1)
    return (True, False, False)       # move left


def backward_policy_as_p1(obs, info=None):
    """Baseline policy for the P1 slot: always move away from the opponent.

    Mirror image of backward_policy_as_p2 but operating on NumpyObsWrapper output
    where positions are already numpy arrays.

    Args:
        obs: Numpy obs dict in P1-perspective format (from NumpyObsWrapper).
        info: Ignored.
    Returns:
        3-tuple (left, right, attack) booleans directing P1 away from P2.
    """
    p1_pos, p2_pos = float(obs["position"][0]), float(obs["position"][1])
    if p1_pos <= p2_pos:
        return (True, False, False)   # move left (further from P2)
    return (False, True, False)       # move right


# ── Agent-as-P2 opponent factory ──────────────────────────────────────────────

def make_agent_p2_opponent(model: PPO):
    """Build a FootsiesEnv opponent callable that runs the trained agent as P2.

    FootsiesEnv passes raw P1-perspective obs to the opponent callback. This factory
    wraps the model so the obs is flipped to P2's frame via swap_to_p2_perspective
    before the model.predict call, ensuring the agent sees a consistent coordinate
    system regardless of which side it occupies.

    Args:
        model: Trained PPO model whose predict method accepts the standard obs dict.
    Returns:
        A callable (obs, info) → action_tuple suitable for FootsiesEnv's opponent= arg.
    """
    def _opponent(obs: dict, info: dict):
        p2_obs = swap_to_p2_perspective(obs)
        action, _ = model.predict(p2_obs, deterministic=True)
        return tuple(bool(a) for a in action)
    return _opponent


# ── Environment factory ───────────────────────────────────────────────────────

def make_eval_env(opponent=None, fast_forward_speed: float = 20.0) -> gym.Env:
    """Construct a single, headless FootsiesEnv wrapped with NumpyObsWrapper.

    Uses sparse reward (dense_reward=False) so the terminal reward sign cleanly
    encodes win (+1) or loss (−1), which is what the match runners check to determine
    the outcome. Wraps in NumpyObsWrapper so model.predict receives typed arrays.

    Args:
        opponent: Callable passed to FootsiesEnv as the P2 controller, or None to
                  use the game's built-in CPU bot.
        fast_forward_speed: Game speed multiplier for the Unity subprocess.
    Returns:
        A NumpyObsWrapper-wrapped FootsiesEnv ready for evaluation.
    """
    env = gym.make(
        "FootsiesEnv-v0",
        game_path=GAME_PATH,
        render_mode=None,
        fast_forward=True,
        fast_forward_speed=fast_forward_speed,
        sync_mode="synced_non_blocking",
        dense_reward=False,      # sparse ±1: clean winner determination from reward sign
        opponent=opponent,
        disable_env_checker=True,
    )
    return NumpyObsWrapper(env)


# ── Match runners ─────────────────────────────────────────────────────────────

def run_match_agent_p1(env: gym.Env, model: PPO) -> dict:
    """Run one match with the trained agent as P1 and return the outcome.

    The P2 controller was baked into the env at creation time via make_eval_env.
    A step cap of MAX_STEPS prevents infinite loops against non-terminating opponents.

    Args:
        env: A NumpyObsWrapper-wrapped FootsiesEnv with the desired P2 opponent set.
        model: Trained PPO model used to select P1 actions deterministically.
    Returns:
        Dict with keys agent_won (bool), agent_guard (int), opp_guard (int),
        frames (int).
    """
    obs, info = env.reset()
    terminated = truncated = False
    steps = 0
    reward = 0.0
    while not (terminated or truncated) and steps < MAX_STEPS:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
    agent_won  = bool(terminated and reward > 0)
    final_guard = info.get("guard", (0, 0))
    return {
        "agent_won":   agent_won,
        "agent_guard": int(final_guard[0]),
        "opp_guard":   int(final_guard[1]),
        "frames":      info.get("frame", steps),
    }


def run_match_agent_p2(env: gym.Env, p1_policy) -> dict:
    """Run one match with the trained agent as P2 and return the outcome.

    The agent controls P2 via the opponent callback baked into the env. This function
    feeds p1_policy actions into env.step() as P1, so reward signs are from P1's
    perspective: a negative terminal reward means P2 (the agent) won.

    Args:
        env: FootsiesEnv where the agent's model was set as env.opponent at creation.
        p1_policy: Callable (obs, info) → action_tuple used to control P1 each step.
    Returns:
        Dict with keys agent_won (bool), agent_guard (int), opp_guard (int),
        frames (int) — all expressed from the agent's (P2) perspective.
    """
    obs, info = env.reset()  # obs is NumpyObsWrapper-processed P1 view
    terminated = truncated = False
    steps = 0
    reward = 0.0
    while not (terminated or truncated) and steps < MAX_STEPS:
        p1_action = p1_policy(obs, info)  # numpy obs is fine for both baselines
        obs, reward, terminated, truncated, info = env.step(p1_action)
        steps += 1
    # P1 reward < 0 means P2 (agent) won
    agent_won  = bool(terminated and reward < 0)
    final_guard = info.get("guard", (0, 0))
    return {
        "agent_won":   agent_won,
        "agent_guard": int(final_guard[1]),   # agent is P2
        "opp_guard":   int(final_guard[0]),
        "frames":      info.get("frame", steps),
    }


# ── Statistics ────────────────────────────────────────────────────────────────

def wilson_ci(wins: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    """Compute the Wilson score 95% confidence interval for a binomial proportion.

    More accurate than the normal approximation near p=0 or p=1, which matters when
    win rates are low. Returns (0, 1) as a fallback when n is 0.

    Args:
        wins: Number of successes (wins).
        n: Total trials (matches).
        z: Standard normal quantile; default 1.959964 ≈ z_{0.975} for 95% CI.
    Returns:
        (lower, upper) confidence interval bounds, both clipped to [0, 1].
    """
    if n == 0:
        return (0.0, 1.0)
    p = wins / n
    denom  = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (round(max(0.0, center - margin), 4), round(min(1.0, center + margin), 4))


def wilcoxon_vs_chance(results: list[bool]) -> tuple[float, float]:
    """Run a one-sample Wilcoxon signed-rank test against H₀: win_rate = 0.5.

    Encodes each result as +1 (win) or −1 (loss) and tests whether the median
    differs from zero. Handles degenerate inputs (all wins or all losses) by
    returning (nan, 0.0) or (nan, 1.0) respectively.

    Args:
        results: List of booleans, one per match (True = agent won).
    Returns:
        (statistic, p_value) floats; either may be nan for degenerate inputs.
    """
    x = np.array([1.0 if w else -1.0 for w in results])
    if len(set(x)) == 1:
        return (float("nan"), 0.0 if x[0] > 0 else 1.0)
    try:
        stat, p = scipy_wilcoxon(x, zero_method="wilcox", correction=False)
        return (float(stat), float(p))
    except ValueError:
        return (float("nan"), float("nan"))


# ── Main evaluation ───────────────────────────────────────────────────────────

def evaluate_baseline(
    baseline_name: str,
    model: PPO,
    n_matches: int,
    fast_forward_speed: float,
    agent_p2_opponent,
    p1_policy,
    cpu_bot: bool,
) -> list[dict]:
    """Play n_matches against one baseline policy and return per-match result rows.

    For non-cpu_bot baselines, n_matches is split evenly: n/2 with the agent as P1
    and n/2 with the agent as P2, ensuring the win-rate estimate is side-agnostic.
    For cpu_bot the game's built-in bot must occupy P2, so all matches are agent=P1.
    A 1.5-second sleep between phases lets the OS release the game's network port.

    Args:
        baseline_name: Label stored in the result rows (e.g. "random").
        model: Trained PPO model used for the agent's actions.
        n_matches: Total matches to play (must be even for non-cpu_bot baselines).
        fast_forward_speed: Game speed multiplier.
        agent_p2_opponent: Pre-built opponent callable for the P2-agent phase.
        p1_policy: Baseline callable used as P1 during the P2-agent phase.
        cpu_bot: If True, skip the P2 phase entirely.
    Returns:
        List of dicts with keys match_id, opponent, agent_side, agent_won,
        agent_guard, opp_guard, frames.
    """
    rows = []
    match_id = 0

    # ── Phase 1: agent as P1 ──────────────────────────────────────────────────
    p2_opponent_fn = None if cpu_bot else p1_policy  # None → --p2-bot in-game
    # For swapped-side baselines, the opponent callable is the baseline policy itself.
    # p1_policy here plays as P2 when opponent= is set; naming is slightly confusing:
    # "p1_policy" means "the policy for the other player", which is P2 in this phase.
    p2_opp = None if cpu_bot else (
        backward_policy_as_p2 if baseline_name == "backward" else random_policy
    )

    n_as_p1 = n_matches if cpu_bot else n_matches // 2
    print(f"  {baseline_name} │ {n_as_p1} matches as P1 …", end="", flush=True)
    env = make_eval_env(opponent=p2_opp, fast_forward_speed=fast_forward_speed)
    try:
        for _ in range(n_as_p1):
            r = run_match_agent_p1(env, model)
            rows.append({"match_id": match_id, "opponent": baseline_name,
                         "agent_side": "P1", **r})
            match_id += 1
    finally:
        env.close()
        time.sleep(1.5)   # let OS release ports
    won_p1 = sum(r["agent_won"] for r in rows)
    print(f" {won_p1}/{n_as_p1} wins")

    if cpu_bot:
        return rows   # skip P2 phase for cpu_bot (built-in bot must be P2)

    # ── Phase 2: agent as P2 ─────────────────────────────────────────────────
    n_as_p2 = n_matches - n_as_p1
    print(f"  {baseline_name} │ {n_as_p2} matches as P2 …", end="", flush=True)
    env = make_eval_env(opponent=agent_p2_opponent, fast_forward_speed=fast_forward_speed)
    try:
        for _ in range(n_as_p2):
            r = run_match_agent_p2(env, p1_policy)
            rows.append({"match_id": match_id, "opponent": baseline_name,
                         "agent_side": "P2", **r})
            match_id += 1
    finally:
        env.close()
        time.sleep(1.5)
    won_p2 = sum(r["agent_won"] for r in rows if r["agent_side"] == "P2")
    print(f" {won_p2}/{n_as_p2} wins")

    return rows


def main():
    """CLI entry point: load a model, run the league evaluation, and write results.

    Iterates over all three baselines (random, backward, cpu_bot), calls
    evaluate_baseline for each, computes Wilson CIs and Wilcoxon tests with
    Bonferroni correction, then writes league_results.csv and league_summary.json
    to the directory specified by --results-dir.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",   default=None,
                        help="Path to .zip model. Default: latest checkpoint in models/")
    parser.add_argument("--matches", type=int, default=100,
                        help="Total matches per baseline (default 100; must be even)")
    parser.add_argument("--speed",   type=float, default=20.0,
                        help="Game fast-forward multiplier")
    parser.add_argument("--results-dir", default=RESULTS_DIR,
                        help="Directory for CSV and JSON output (default: results/)")
    args = parser.parse_args()

    if args.matches % 2 != 0:
        parser.error("--matches must be even (split 50/50 between P1 and P2)")

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = args.model
    if model_path is None:
        zips = sorted(glob.glob(os.path.join("models", "*.zip")))
        if not zips:
            raise FileNotFoundError("No .zip checkpoints found in models/")
        model_path = zips[-1]
    print(f"Model: {model_path}")
    model = PPO.load(model_path)

    agent_p2_opponent = make_agent_p2_opponent(model)

    baselines = [
        ("random",   random_policy,          False),
        ("backward", backward_policy_as_p1,  False),
        ("cpu_bot",  None,                   True),
    ]

    results_dir = args.results_dir
    os.makedirs(results_dir, exist_ok=True)
    csv_path  = os.path.join(results_dir, "league_results.csv")
    json_path = os.path.join(results_dir, "league_summary.json")

    CSV_FIELDS = ["match_id", "opponent", "agent_side",
                  "agent_won", "agent_guard", "opp_guard", "frames"]
    all_rows: list[dict] = []
    summary: dict = {}

    for baseline_name, p1_policy, cpu_bot in baselines:
        print(f"\n── {baseline_name.upper()} ──")
        rows = evaluate_baseline(
            baseline_name=baseline_name,
            model=model,
            n_matches=args.matches,
            fast_forward_speed=args.speed,
            agent_p2_opponent=agent_p2_opponent,
            p1_policy=p1_policy,
            cpu_bot=cpu_bot,
        )
        all_rows.extend(rows)

        wins   = [r["agent_won"] for r in rows]
        n      = len(wins)
        n_wins = sum(wins)
        win_rate     = n_wins / n
        ci_lo, ci_hi = wilson_ci(n_wins, n)
        w_stat, w_p  = wilcoxon_vs_chance(wins)

        summary[baseline_name] = {
            "n_matches":           n,
            "wins":                n_wins,
            "win_rate":            round(win_rate, 4),
            "wilson_ci_95":        [ci_lo, ci_hi],
            "wilcoxon_stat":       round(w_stat, 4) if not math.isnan(w_stat) else None,
            "wilcoxon_p":          round(w_p, 6)    if not math.isnan(w_p)    else None,
            "bonferroni_alpha":    round(ALPHA, 5),
            "significant":         (not math.isnan(w_p)) and w_p < ALPHA,
            "mean_agent_guard":    round(np.mean([r["agent_guard"] for r in rows]), 3),
            "mean_opp_guard":      round(np.mean([r["opp_guard"]   for r in rows]), 3),
            "mean_frames":         round(np.mean([r["frames"]       for r in rows]), 1),
            "note_cpu_bot":        "all 100 matches played as P1 (built-in bot must be P2)"
                                   if cpu_bot else None,
        }
        print(f"  win rate: {win_rate:.1%}  CI95: [{ci_lo:.3f}, {ci_hi:.3f}]"
              f"  Wilcoxon p={w_p:.4f}  sig@{ALPHA:.4f}: {summary[baseline_name]['significant']}")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(all_rows)
    print(f"\nResults CSV  → {csv_path}")

    # ── Write JSON ────────────────────────────────────────────────────────────
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary JSON → {json_path}")


if __name__ == "__main__":
    main()
