"""
Tactical scenario evaluation for a trained FOOTSIES PPO agent.

Reads scenarios.json (nested agent/opponent format), constructs the obs dict for
each scenario, and records whether the model's deterministic action matches the
expected semantic action. No game process is needed — the model only ever sees the
obs dict, so model.predict(synthetic_obs) is identical to querying it mid-game in
that exact state.

Expected scenario format (see scenarios.json for full example)
--------------------------------------------------------------
{
  "metadata": { "action_map": {...}, "move_map": {...} },
  "scenarios": [
    {
      "id": "WP-01",
      "category": "whiff_punishment",
      "description": "...",
      "agent":    { "side": "P1", "position": -1.0, "guard": 3,
                    "move": "STAND", "move_frame": 0 },
      "opponent": {                  "position":  0.8, "guard": 3,
                    "move": "B_ATTACK", "move_frame": 20 },
      "expected_action": "FORWARD_ATTACK"
    }
  ]
}

Action strings (resolved against agent.side)
  IDLE / FORWARD / BACKWARD / ATTACK / FORWARD_ATTACK / BACKWARD_ATTACK

Outputs
-------
  results/scenario_results.csv   – per-scenario pass/fail
  (stdout)                       – aggregate pass rate by category
"""
import os
import csv
import json
import argparse
import glob
import numpy as np

from stable_baselines3 import PPO
from footsies_gym.moves import FootsiesMove

RESULTS_DIR    = os.path.join(os.path.dirname(__file__), "results")
SCENARIOS_FILE = os.path.join(os.path.dirname(__file__), "scenarios.json")
CSV_PATH       = os.path.join(RESULTS_DIR, "scenario_results.csv")

CSV_FIELDS = [
    "id", "category", "description", "agent_side",
    "p1_guard", "p2_guard", "p1_position", "p2_position",
    "p1_move", "p2_move", "p1_move_frame", "p2_move_frame",
    "expected_semantic", "expected_action", "model_action", "passed",
]

# Move name → obs index (env excludes WIN and DEAD from the space)
_RELEVANT_MOVES = [m for m in FootsiesMove if m not in {FootsiesMove.WIN, FootsiesMove.DEAD}]
MOVE_NAME_TO_INDEX: dict[str, int] = {m.name: i for i, m in enumerate(_RELEVANT_MOVES)}

# Semantic action → MultiBinary(3) [left, right, attack] by agent side
ACTION_TABLE: dict[str, dict[str, list[int]]] = {
    "P1": {
        "IDLE":             [0, 0, 0],
        "FORWARD":          [0, 1, 0],
        "BACKWARD":         [1, 0, 0],
        "ATTACK":           [0, 0, 1],
        "FORWARD_ATTACK":   [0, 1, 1],
        "BACKWARD_ATTACK":  [1, 0, 1],
    },
    "P2": {
        "IDLE":             [0, 0, 0],
        "FORWARD":          [1, 0, 0],   # mirrored
        "BACKWARD":         [0, 1, 0],
        "ATTACK":           [0, 0, 1],
        "FORWARD_ATTACK":   [1, 0, 1],
        "BACKWARD_ATTACK":  [0, 1, 1],
    },
}


def resolve_action(semantic: str, side: str) -> list[int]:
    """Convert a semantic action string to MultiBinary(3) for the given agent side."""
    key = semantic.upper()
    table = ACTION_TABLE.get(side.upper(), ACTION_TABLE["P1"])
    if key not in table:
        raise ValueError(
            f"Unknown action '{semantic}'. Valid: {list(table)}"
        )
    return table[key]


def build_obs(sc: dict) -> tuple[dict, dict]:
    """
    Build the obs dict from a scenario (always from agent's perspective as P1).
    Returns (obs, debug_info) where debug_info records the raw field values.

    Agent perspective:
      obs["guard"][0]      = agent guard
      obs["position"][0]   = agent position (mirrored for P2 side)
      obs["guard"][1]      = opponent guard
      obs["position"][1]   = opponent position (mirrored for P2 side)
    """
    agent    = sc["agent"]
    opponent = sc["opponent"]
    side     = agent.get("side", "P1").upper()

    # Positions: present to the model in the same coordinate frame as training (P1=left).
    # For P2 agents, mirror (negate) both positions so the agent sees itself at negative X.
    if side == "P1":
        ag_pos   = float(agent["position"])
        opp_pos  = float(opponent["position"])
    else:
        ag_pos   = -float(agent["position"])
        opp_pos  = -float(opponent["position"])

    ag_move  = agent["move"].upper()
    opp_move = opponent["move"].upper()
    if ag_move not in MOVE_NAME_TO_INDEX:
        raise ValueError(f"Unknown agent move '{ag_move}'. Valid: {list(MOVE_NAME_TO_INDEX)}")
    if opp_move not in MOVE_NAME_TO_INDEX:
        raise ValueError(f"Unknown opponent move '{opp_move}'. Valid: {list(MOVE_NAME_TO_INDEX)}")

    obs = {
        "guard":      np.array([int(agent["guard"]),              int(opponent["guard"])],       dtype=np.int64),
        "move":       np.array([MOVE_NAME_TO_INDEX[ag_move],     MOVE_NAME_TO_INDEX[opp_move]], dtype=np.int64),
        "move_frame": np.array([float(agent["move_frame"]),      float(opponent["move_frame"])], dtype=np.float32),
        "position":   np.array([ag_pos,                          opp_pos],                       dtype=np.float32),
    }

    debug = {
        "agent_side":    side,
        "p1_guard":      int(agent["guard"]),
        "p2_guard":      int(opponent["guard"]),
        "p1_position":   ag_pos,
        "p2_position":   opp_pos,
        "p1_move":       ag_move,
        "p2_move":       opp_move,
        "p1_move_frame": int(agent["move_frame"]),
        "p2_move_frame": int(opponent["move_frame"]),
    }
    return obs, debug


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None,
                        help="Path to .zip model. Default: latest checkpoint in models/")
    parser.add_argument("--scenarios", default=SCENARIOS_FILE,
                        help=f"Path to scenarios JSON (default: {SCENARIOS_FILE})")
    args = parser.parse_args()

    # ── Load model ────────────────────────────────────────────────────────────
    model_path = args.model
    if model_path is None:
        zips = sorted(glob.glob(os.path.join("models", "*.zip")))
        if not zips:
            raise FileNotFoundError("No .zip checkpoints found in models/")
        model_path = zips[-1]
    print(f"Model:     {model_path}")

    model = PPO.load(model_path)

    # ── Load scenarios ────────────────────────────────────────────────────────
    if not os.path.exists(args.scenarios):
        raise FileNotFoundError(
            f"scenarios.json not found at {args.scenarios}."
        )
    with open(args.scenarios) as f:
        data = json.load(f)
    scenarios = data["scenarios"]
    print(f"Scenarios: {len(scenarios)} loaded\n")

    os.makedirs(RESULTS_DIR, exist_ok=True)

    rows: list[dict] = []
    by_cat: dict[str, list[bool]] = {}

    for sc in scenarios:
        obs, debug = build_obs(sc)
        side = debug["agent_side"]

        expected_semantic = sc["expected_action"].upper()
        expected          = resolve_action(expected_semantic, side)

        action, _ = model.predict(obs, deterministic=True)
        actual    = [int(a) for a in action]
        passed    = (actual == expected)

        cat = sc.get("category", "uncategorised")
        by_cat.setdefault(cat, []).append(passed)

        rows.append({
            "id":               sc["id"],
            "category":         cat,
            "description":      sc.get("description", ""),
            "agent_side":       side,
            **debug,
            "expected_semantic": expected_semantic,
            "expected_action":  str(expected),
            "model_action":     str(actual),
            "passed":           passed,
        })

        status = "PASS ✓" if passed else "FAIL ✗"
        print(f"  [{status}] {sc['id']:8s} [{cat}]  "
              f"exp={expected}  got={actual}  | {sc.get('description','')[:50]}")

    # ── Per-category summary ──────────────────────────────────────────────────
    print("\n── Category summary ──────────────────────────────────────────────")
    for cat in sorted(by_cat):
        results = by_cat[cat]
        n, n_ok = len(results), sum(results)
        bar = "█" * n_ok + "░" * (n - n_ok)
        print(f"  {cat:22s}: {n_ok:2}/{n:2}  {bar}")

    total     = len(rows)
    total_pass = sum(r["passed"] for r in rows)
    print(f"\n  Overall: {total_pass}/{total} ({100*total_pass/total:.0f}%)")

    # ── Write CSV ─────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nScenario CSV → {CSV_PATH}")


if __name__ == "__main__":
    main()
