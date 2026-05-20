"""
generate_thesis_figures.py

Reads pre-computed result files and writes publication-ready PNGs to figures/.

Inputs (all in results/):
    training_curves.csv, league_results.csv,
    league_summary.json, scenario_results.csv

Outputs (all in figures/):
    training_reward.png, training_length.png, training_combined.png,
    league_winrates.png, side_bias.png,
    scenario_pass_rates.png, action_distribution.png

Usage:
    python generate_thesis_figures.py
"""

import ast
import collections
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

# ── Style ─────────────────────────────────────────────────────────────────────
matplotlib.rcParams.update(
    {
        # Times New Roman first; Liberation Serif is a metric-compatible
        # open-source substitute available on most Linux systems
        "font.family":          "serif",
        "font.serif":           ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "font.size":            10,
        "axes.titlesize":       11,
        "axes.titlepad":        6,
        "axes.labelsize":       10,
        "axes.labelpad":        4,
        "xtick.labelsize":      9,
        "ytick.labelsize":      9,
        "legend.fontsize":      9,
        "legend.frameon":       False,
        "axes.spines.top":      False,
        "axes.spines.right":    False,
        "axes.grid":            True,
        "grid.linewidth":       0.4,
        "grid.alpha":           0.4,
        "figure.dpi":           200,
        "savefig.dpi":          200,
        "savefig.transparent":  True,
        "savefig.bbox":         "tight",
    }
)

# ~16 cm wide in inches
FIG_W = 6.30
DPI = 200

# Colour palette
C_BLUE   = "#2C6FAC"
C_RED    = "#C0392B"
C_GREEN  = "#27AE60"
C_PURPLE = "#8E44AD"
C_GRAY   = "#7F7F7F"

RUN_COLORS  = {"PPO_footsies_1": C_BLUE,   "PPO_footsies_2": C_RED}
RUN_LABELS  = {"PPO_footsies_1": "Run 1 (0–550 k steps)",
               "PPO_footsies_2": "Run 2 (0–1 M steps)"}

OPP_COLORS  = {"random": C_BLUE, "backward": C_GREEN, "cpu_bot": C_RED}
OPP_LABELS  = {"random": "Random", "backward": "Backward", "cpu_bot": "CPU Bot"}

CAT_COLORS  = {
    "guard_pressure":   C_PURPLE,
    "positional":       C_BLUE,
    "special_defense":  C_GREEN,
    "whiff_punishment": C_RED,
}

# P1 bit-vector → semantic action name
ACTION_MAP_P1 = {
    (0, 0, 0): "Idle",
    (0, 0, 1): "Attack",
    (0, 1, 0): "Forward",
    (0, 1, 1): "Fwd + Attack",
    (1, 0, 0): "Backward",
    (1, 0, 1): "Bwd + Attack",
}
# P2 perspective flips left/right
ACTION_MAP_P2 = {
    (0, 0, 0): "Idle",
    (0, 0, 1): "Attack",
    (1, 0, 0): "Forward",
    (1, 0, 1): "Fwd + Attack",
    (0, 1, 0): "Backward",
    (0, 1, 1): "Bwd + Attack",
}

RESULTS_DIR = "results"
FIGURES_DIR = "figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def save(fig: plt.Figure, filename: str) -> None:
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


def smooth(series: pd.Series, window: int = 20) -> pd.Series:
    """Centred rolling mean that handles short series gracefully."""
    return series.rolling(window=window, center=True, min_periods=1).mean()


def wilson_ci(wins: int, n: int, z: float = 1.96):
    """Return (lo, hi) Wilson score 95% confidence interval."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    denom   = 1.0 + z ** 2 / n
    centre  = (p + z ** 2 / (2 * n)) / denom
    margin  = z * (p * (1 - p) / n + z ** 2 / (4 * n ** 2)) ** 0.5 / denom
    return max(0.0, centre - margin), min(1.0, centre + margin)


def pct_fmt(ax):
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=0))


def main():
    """Load result files and generate all 7 thesis figures into figures/.

    Reads training_curves.csv, league_results.csv, league_summary.json, and
    scenario_results.csv from RESULTS_DIR, then calls the per-figure plotting
    blocks in sequence. All I/O is confined here so the module can be imported
    by Sphinx autodoc without touching the filesystem.
    """
    # ── Load data ─────────────────────────────────────────────────────────────
    print("Loading data …")
    train_df = pd.read_csv(os.path.join(RESULTS_DIR, "training_curves.csv"))
    league_df = pd.read_csv(os.path.join(RESULTS_DIR, "league_results.csv"))
    league_df["agent_won"] = league_df["agent_won"].map(
        {"True": True, "False": False, True: True, False: False}
    ).astype(bool)
    scenarios_df = pd.read_csv(os.path.join(RESULTS_DIR, "scenario_results.csv"))
    scenarios_df["passed"] = scenarios_df["passed"].map(
        {"True": True, "False": False, True: True, False: False}
    ).astype(bool)
    with open(os.path.join(RESULTS_DIR, "league_summary.json")) as fh:
        summary = json.load(fh)

    opponents = list(summary.keys())

    # ── 1. training_reward.png ────────────────────────────────────────────────
    print("\n[1/7] training_reward.png")
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    for run, grp in train_df.groupby("run"):
        grp = grp.sort_values("step")
        col = RUN_COLORS.get(run, C_GRAY)
        ax.plot(grp["step"] / 1e6, grp["ep_rew_mean"],
                alpha=0.18, linewidth=0.8, color=col)
        ax.plot(grp["step"] / 1e6, smooth(grp["ep_rew_mean"]),
                linewidth=1.8, color=col, label=RUN_LABELS.get(run, run))
    ax.axhline(0, color="black", linewidth=0.7, linestyle="--", alpha=0.45, zorder=0)
    ax.set_xlabel("Training steps (×10⁶)")
    ax.set_ylabel("Mean episode reward")
    ax.set_title("Episode Reward During Training")
    ax.legend()
    fig.tight_layout()
    save(fig, "training_reward.png")

    # ── 2. training_length.png ────────────────────────────────────────────────
    print("[2/7] training_length.png")
    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    for run, grp in train_df.groupby("run"):
        grp = grp.sort_values("step")
        col = RUN_COLORS.get(run, C_GRAY)
        ax.plot(grp["step"] / 1e6, grp["ep_len_mean"],
                alpha=0.18, linewidth=0.8, color=col)
        ax.plot(grp["step"] / 1e6, smooth(grp["ep_len_mean"]),
                linewidth=1.8, color=col, label=RUN_LABELS.get(run, run))
    ax.set_xlabel("Training steps (×10⁶)")
    ax.set_ylabel("Mean episode length (frames)")
    ax.set_title("Episode Length During Training")
    ax.legend()
    fig.tight_layout()
    save(fig, "training_length.png")

    # ── 3. training_combined.png ──────────────────────────────────────────────
    print("[3/7] training_combined.png")
    PANELS = [
        ("ep_rew_mean",        "Mean episode reward",          "Episode Reward"),
        ("ep_len_mean",        "Mean episode length (frames)", "Episode Length"),
        ("value_loss",         "Value loss",                   "Value Loss"),
        ("explained_variance", "Explained variance",           "Explained Variance"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(FIG_W, 5.4))
    for ax, (col, ylabel, title) in zip(axes.flatten(), PANELS):
        for run, grp in train_df.groupby("run"):
            grp = grp.sort_values("step").dropna(subset=[col])
            c = RUN_COLORS.get(run, C_GRAY)
            ax.plot(grp["step"] / 1e6, grp[col],
                    alpha=0.15, linewidth=0.7, color=c)
            ax.plot(grp["step"] / 1e6, smooth(grp[col]),
                    linewidth=1.4, color=c, label=RUN_LABELS.get(run, run))
        ax.set_title(title)
        ax.set_xlabel("Steps (×10⁶)", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    handles = [
        Line2D([0], [0], color=C_BLUE, linewidth=1.6, label="Run 1 (0–550 k)"),
        Line2D([0], [0], color=C_RED,  linewidth=1.6, label="Run 2 (0–1 M)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.01), fontsize=9)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    save(fig, "training_combined.png")

    # ── 4. league_winrates.png ────────────────────────────────────────────────
    print("[4/7] league_winrates.png")
    wr_vals = [summary[o]["win_rate"]     for o in opponents]
    ci_vals = [summary[o]["wilson_ci_95"] for o in opponents]
    p_vals  = [summary[o]["wilcoxon_p"]   for o in opponents]
    sig     = [summary[o]["significant"]  for o in opponents]
    err_lo  = [wr - ci[0] for wr, ci in zip(wr_vals, ci_vals)]
    err_hi  = [ci[1] - wr for wr, ci in zip(wr_vals, ci_vals)]

    fig, ax = plt.subplots(figsize=(FIG_W, 3.8))
    x = np.arange(len(opponents))
    bar_cols = [OPP_COLORS.get(o, C_GRAY) for o in opponents]
    ax.bar(x, wr_vals, color=bar_cols, width=0.52, zorder=3,
           yerr=[err_lo, err_hi], capsize=6,
           error_kw=dict(elinewidth=1.3, ecolor="black", capthick=1.3))
    ax.axhline(0.5, color="black", linewidth=0.9, linestyle="--",
               alpha=0.5, label="Chance level (50 %)", zorder=2)
    for xi, (wr, p, s, ehi) in enumerate(zip(wr_vals, p_vals, sig, err_hi)):
        stars = " *" if s else ""
        ax.text(xi, wr + ehi + 0.04, f"p = {p:.4f}{stars}",
                ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([OPP_LABELS.get(o, o) for o in opponents])
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("League Win Rates vs. Baselines  (95 % Wilson CI, Bonferroni-corrected)")
    ax.legend(loc="upper left")
    pct_fmt(ax)
    fig.tight_layout()
    save(fig, "league_winrates.png")

    # ── 5. side_bias.png ──────────────────────────────────────────────────────
    print("[5/7] side_bias.png")
    fig, ax = plt.subplots(figsize=(FIG_W, 3.8))
    width = 0.32
    x = np.arange(len(opponents))
    side_colors = {"P1": C_BLUE, "P2": C_RED}
    for i, side in enumerate(["P1", "P2"]):
        rates, err_los, err_his = [], [], []
        for opp in opponents:
            sub = league_df[(league_df["opponent"] == opp) &
                            (league_df["agent_side"] == side)]
            if len(sub) == 0:
                rates.append(np.nan); err_los.append(0); err_his.append(0)
            else:
                wins = sub["agent_won"].sum()
                n    = len(sub)
                lo, hi = wilson_ci(wins, n)
                rate = wins / n
                rates.append(rate)
                err_los.append(rate - lo)
                err_his.append(hi - rate)
        offset = (i - 0.5) * width
        valid  = [(j, r) for j, r in enumerate(rates) if not np.isnan(r)]
        if valid:
            xs_v = np.array([j for j, _ in valid])
            rs_v = np.array([r for _, r in valid])
            lo_v = np.array([err_los[j] for j, _ in valid])
            hi_v = np.array([err_his[j] for j, _ in valid])
            ax.bar(xs_v + offset, rs_v, width=width,
                   color=side_colors[side], alpha=0.87,
                   label=f"Agent as {side}", zorder=3,
                   yerr=[lo_v, hi_v], capsize=5,
                   error_kw=dict(elinewidth=1.2, ecolor="black", capthick=1.2))
        for j, r in enumerate(rates):
            if np.isnan(r):
                ax.text(j + offset, 0.025, "N/A", ha="center", va="bottom",
                        fontsize=8, color=C_GRAY, style="italic")
    ax.axhline(0.5, color="black", linewidth=0.9, linestyle="--", alpha=0.5, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels([OPP_LABELS.get(o, o) for o in opponents])
    ax.set_ylabel("Win rate")
    ax.set_ylim(0, 1.05)
    ax.set_title("Win Rate by Agent Side (P1 vs P2)")
    ax.legend()
    pct_fmt(ax)
    fig.tight_layout()
    save(fig, "side_bias.png")

    # ── 6. scenario_pass_rates.png ────────────────────────────────────────────
    print("[6/7] scenario_pass_rates.png")
    by_cat = (
        scenarios_df.groupby("category")["passed"]
        .agg(passed="sum", total="count")
        .reset_index()
    )
    by_cat["rate"] = by_cat["passed"] / by_cat["total"]
    cis = [wilson_ci(r["passed"], r["total"]) for _, r in by_cat.iterrows()]
    by_cat["ci_lo"] = [r - lo for (lo, _), r in zip(cis, by_cat["rate"])]
    by_cat["ci_hi"] = [hi - r for (_, hi), r in zip(cis, by_cat["rate"])]
    cat_order = sorted(by_cat["category"].tolist())
    by_cat    = by_cat.set_index("category").loc[cat_order].reset_index()
    cols_cat  = [CAT_COLORS.get(c, C_GRAY) for c in by_cat["category"]]

    fig, ax = plt.subplots(figsize=(FIG_W, 3.8))
    x = np.arange(len(by_cat))
    ax.bar(x, by_cat["rate"], color=cols_cat, width=0.52, zorder=3,
           yerr=[by_cat["ci_lo"].values, by_cat["ci_hi"].values], capsize=6,
           error_kw=dict(elinewidth=1.3, ecolor="black", capthick=1.3))
    for xi, row in by_cat.iterrows():
        y_top = row["rate"] + row["ci_hi"] + 0.04
        ax.text(xi, y_top, f"{int(row['passed'])}/{int(row['total'])}",
                ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in by_cat["category"]], fontsize=9
    )
    ax.set_ylabel("Pass rate")
    ax.set_ylim(0, 1.18)
    ax.set_title("Tactical Scenario Pass Rates by Category  (95 % Wilson CI)")
    pct_fmt(ax)
    fig.tight_layout()
    save(fig, "scenario_pass_rates.png")

    # ── 7. action_distribution.png ────────────────────────────────────────────
    print("[7/7] action_distribution.png")
    semantic_counts: collections.Counter = collections.Counter()
    for _, row in scenarios_df.iterrows():
        tup   = tuple(ast.literal_eval(row["model_action"]))
        amap  = ACTION_MAP_P2 if row["agent_side"] == "P2" else ACTION_MAP_P1
        label = amap.get(tup, str(tup))
        semantic_counts[label] += 1

    ACT_ORDER = ["Idle", "Forward", "Backward", "Attack", "Fwd + Attack", "Bwd + Attack"]
    freqs   = [semantic_counts.get(a, 0) for a in ACT_ORDER]
    n_total = sum(freqs)

    fig, ax = plt.subplots(figsize=(FIG_W, 3.6))
    bars = ax.bar(ACT_ORDER, freqs, color=C_BLUE, width=0.55, zorder=3)
    for bar, f in zip(bars, freqs):
        pct = 100 * f / n_total if n_total else 0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.15,
            f"{f}  ({pct:.0f} %)",
            ha="center", va="bottom", fontsize=8.5,
        )
    ax.set_ylabel(f"Count  (n = {n_total} scenarios)")
    ax.set_ylim(0, max(freqs) * 1.25 if freqs else 1)
    ax.set_title("Predicted Action Distribution Across Scenarios")
    fig.tight_layout()
    save(fig, "action_distribution.png")

    print("\nDone — all 7 figures written to figures/")


if __name__ == "__main__":
    main()
