"""
Export TensorBoard scalar logs to CSV.
Reads all PPO_footsies_* subdirectories under logs/ and merges them into
results/training_curves.csv with columns:
  step, ep_rew_mean, ep_len_mean, value_loss, policy_loss,
  entropy_loss, explained_variance, approx_kl, run
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
