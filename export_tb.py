"""
Export TensorBoard scalar logs to CSV.

Reads all ``PPO_footsies_*`` subdirectories under ``logs/`` and merges them into
``results/training_curves.csv`` with columns:
``run``, ``step``, ``ep_rew_mean``, ``ep_len_mean``, ``value_loss``,
``policy_loss``, ``entropy_loss``, ``explained_variance``, ``approx_kl``.
"""
import os
import glob
import csv
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

LOGS_DIR   = os.path.join(os.path.dirname(__file__), "logs")
OUTPUT_CSV = os.path.join(os.path.dirname(__file__), "results", "training_curves.csv")

SCALAR_MAP = {
    "rollout/ep_rew_mean":        "ep_rew_mean",
    "rollout/ep_len_mean":        "ep_len_mean",
    "train/value_loss":           "value_loss",
    "train/policy_gradient_loss": "policy_loss",
    "train/entropy_loss":         "entropy_loss",
    "train/explained_variance":   "explained_variance",
    "train/approx_kl":            "approx_kl",
}
COLUMNS = ["run", "step"] + list(SCALAR_MAP.values())


def load_run(run_dir: str) -> list[dict]:
    """Read all tracked scalars from one TensorBoard event directory.

    Uses EventAccumulator to parse the binary protobuf event files. For each scalar
    tag that appears in SCALAR_MAP, extracts every (step, value) pair and merges them
    into a step-keyed dict so that all scalars recorded at the same step end up in a
    single output row. Tags absent from the run (e.g., value_loss before the first
    gradient update) are silently skipped.

    Args:
        run_dir: Path to a directory containing TensorBoard event files, typically
                 logs/PPO_footsies_N/.
    Returns:
        List of dicts sorted by step, each containing "run", "step", and one key per
        scalar that was recorded at that step. Missing scalars produce no key (the CSV
        writer leaves the cell blank).
    """
    ea = EventAccumulator(run_dir)
    ea.Reload()
    available = set(ea.Tags().get("scalars", []))

    # Build step→values dict; all scalars share the same step axis in SB3
    step_data: dict[int, dict] = {}
    for tb_tag, col in SCALAR_MAP.items():
        if tb_tag not in available:
            continue
        for event in ea.Scalars(tb_tag):
            row = step_data.setdefault(event.step, {})
            row[col] = event.value

    rows = []
    run_name = os.path.basename(run_dir.rstrip("/"))
    for step in sorted(step_data):
        row = {"run": run_name, "step": step}
        row.update(step_data[step])
        rows.append(row)
    return rows


def main():
    """Discover all PPO training runs under logs/, export their scalars to one CSV.

    Scans for PPO_footsies_*/ subdirectories in LOGS_DIR, calls load_run for each,
    concatenates the rows, and writes them to OUTPUT_CSV. The "run" column in the CSV
    identifies which training run each row belongs to, enabling multi-run comparisons
    in generate_thesis_figures.py.
    """
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

    all_rows: list[dict] = []
    run_dirs = sorted(glob.glob(os.path.join(LOGS_DIR, "PPO_footsies_*/")))
    if not run_dirs:
        raise FileNotFoundError(f"No PPO_footsies_* directories found under {LOGS_DIR}")

    for run_dir in run_dirs:
        rows = load_run(run_dir)
        print(f"  {os.path.basename(run_dir.rstrip('/'))}: {len(rows)} steps "
              f"({rows[0]['step']}..{rows[-1]['step']})" if rows else f"  {run_dir}: empty")
        all_rows.extend(rows)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in all_rows:
            writer.writerow(row)

    print(f"\nExported {len(all_rows)} rows → {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
