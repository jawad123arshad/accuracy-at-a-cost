"""
Entry point for the pilot study.

Usage (from the pilot/ folder):
    python dataset.py            # one-time: scan CodeNet subset + build cache (can also be skipped, run_experiments does it automatically)
    python run_experiments.py    # trains all architectures x regimes, writes outputs/results.csv and outputs/efficiency_frontier.png

Regimes per architecture:
    single_classification -> encoder + classification head only
    single_verdict        -> encoder + verdict head only
    multitask              -> shared encoder + both heads, joint loss

This directly implements C1 (multitask x architecture interaction), using the
C2 measurement protocol (efficiency.py) and produces the C3 scatter plot.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from dataset import prepare_and_cache
from train import train_one


def main():
    print(f"Device: {C.DEVICE}")
    payload = prepare_and_cache()
    splits = payload["splits"]
    vocab_size = payload["vocab_size"]
    num_classes = payload["num_classes"]
    pad_id = payload["stoi"][C.PAD_TOKEN]

    print(f"Subset: {num_classes} problem classes, vocab={vocab_size}, "
          f"train/val/test = {len(splits['train'])}/{len(splits['val'])}/{len(splits['test'])}")

    regimes = [
        ("single_classification", ["classification"]),
        ("single_verdict", ["verdict"]),
        ("multitask", ["classification", "verdict"]),
    ]

    all_results = []
    for arch in C.ARCHITECTURES:
        for regime_name, tasks in regimes:
            for seed in C.SEEDS:
                print(f"\n=== {arch} :: {regime_name} :: seed={seed} ===")
                result, _ = train_one(
                    arch=arch, tasks=tasks, vocab_size=vocab_size, num_classes=num_classes,
                    pad_id=pad_id, splits=splits, seed=seed,
                )
                result["regime"] = regime_name
                all_results.append(result)
                print(f"  -> {result}")

    df = pd.DataFrame(all_results)
    df.to_csv(C.RESULTS_CSV, index=False)
    print(f"\nSaved per-seed results table -> {C.RESULTS_CSV}")

    agg_df = aggregate_seeds(df)
    agg_path = C.RESULTS_CSV.replace("results.csv", "results_aggregated.csv")
    agg_df.to_csv(agg_path, index=False)
    print(f"Saved aggregated (mean +/- std over {len(C.SEEDS)} seeds) -> {agg_path}")
    print(agg_df.to_string(index=False))

    compute_transfer_deltas(agg_df, df)
    make_efficiency_plot(agg_df)


def aggregate_seeds(df):
    """Collapses per-seed rows into mean +/- std per (arch, regime)."""
    metric_cols = [c for c in df.columns
                   if c not in ("arch", "regime", "tasks", "seed", "best_epoch")]
    grouped = df.groupby(["arch", "regime", "tasks"])
    mean_df = grouped[metric_cols].mean().add_suffix("_mean")
    std_df = grouped[metric_cols].std().add_suffix("_std")
    out = pd.concat([mean_df, std_df], axis=1).reset_index()
    return out.sort_values(["arch", "regime"])


def compute_transfer_deltas(agg_df, per_seed_df):
    """C1's headline number: single-task vs multitask accuracy delta, per
    architecture and per task, using seed-aggregated means, plus a simple
    Welch's t-test flag so you don't report a delta that's just seed noise."""
    from scipy import stats

    rows = []
    for arch in C.ARCHITECTURES:
        sub_agg = agg_df[agg_df["arch"] == arch]
        sub_raw = per_seed_df[per_seed_df["arch"] == arch]
        mt_agg = sub_agg[sub_agg["regime"] == "multitask"].iloc[0]
        for task, single_regime in [("classification", "single_classification"), ("verdict", "single_verdict")]:
            single_agg = sub_agg[sub_agg["regime"] == single_regime].iloc[0]
            delta = mt_agg[f"test_acc_{task}_mean"] - single_agg[f"test_acc_{task}_mean"]

            mt_vals = sub_raw[sub_raw["regime"] == "multitask"][f"test_acc_{task}"].values
            single_vals = sub_raw[sub_raw["regime"] == single_regime][f"test_acc_{task}"].values
            if len(mt_vals) > 1 and len(single_vals) > 1:
                _, p_value = stats.ttest_ind(mt_vals, single_vals, equal_var=False)
            else:
                p_value = float("nan")

            rows.append({
                "arch": arch, "task": task,
                "single_task_acc_mean": round(single_agg[f"test_acc_{task}_mean"], 4),
                "single_task_acc_std": round(single_agg[f"test_acc_{task}_std"], 4),
                "multitask_acc_mean": round(mt_agg[f"test_acc_{task}_mean"], 4),
                "multitask_acc_std": round(mt_agg[f"test_acc_{task}_std"], 4),
                "delta": round(delta, 4),
                "p_value": round(p_value, 4) if p_value == p_value else float("nan"),
                "significant_p<0.05": bool(p_value < 0.05) if p_value == p_value else False,
            })
    deltas = pd.DataFrame(rows)
    out_path = C.RESULTS_CSV.replace("results.csv", "transfer_deltas.csv")
    deltas.to_csv(out_path, index=False)
    print(f"\nSingle-task vs multitask accuracy deltas, {len(C.SEEDS)}-seed mean +/- std "
          f"(this is your C1 headline table -- only trust deltas flagged significant):")
    print(deltas.to_string(index=False))
    print(f"Saved -> {out_path}")


def make_efficiency_plot(agg_df):
    """C3: descriptive (accuracy, compute-cost) scatter, one point per
    architecture x regime (seed-mean), colored by architecture, marker by regime."""
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"bigru": "tab:blue", "transformer": "tab:orange", "mamba": "tab:green"}
    markers = {"single_classification": "o", "single_verdict": "s", "multitask": "^"}

    for _, row in agg_df.iterrows():
        acc_cols = [c for c in agg_df.columns if c.startswith("test_acc_") and c.endswith("_mean")]
        avg_acc = row[acc_cols].mean(skipna=True)
        ax.scatter(
            row["train_time_sec_mean"], avg_acc,
            c=colors.get(row["arch"], "gray"),
            marker=markers.get(row["regime"], "x"),
            s=90, edgecolors="black", linewidths=0.5,
        )
        ax.annotate(f"{row['arch']}/{row['regime']}", (row["train_time_sec_mean"], avg_acc),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("Training time (s)  [compute cost proxy]")
    ax.set_ylabel("Mean test accuracy across active tasks")
    ax.set_title("Efficiency-Performance Comparison (C3, descriptive)")
    ax.grid(alpha=0.3)

    # legend for architecture colors
    from matplotlib.lines import Line2D
    legend_elems = [Line2D([0], [0], marker="o", color="w", markerfacecolor=c, label=a, markersize=8)
                     for a, c in colors.items()]
    ax.legend(handles=legend_elems, title="architecture", loc="best")

    fig.tight_layout()
    fig.savefig(C.PLOT_PATH, dpi=150)
    print(f"Saved efficiency plot -> {C.PLOT_PATH}")


if __name__ == "__main__":
    main()
