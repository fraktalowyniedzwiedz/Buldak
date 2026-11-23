import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_seen_vs_unseen_metrics(metrics_df, fig_dir):
    summary = metrics_df.groupby(["seen", "mode"], as_index=False).agg(
        seen_RMSE=("seen_RMSE", "mean"),
        unseen_RMSE=("unseen_RMSE", "mean"),
        seen_MAE=("seen_MAE", "mean"),
        unseen_MAE=("unseen_MAE", "mean"),
    )
    for metric in ("RMSE", "MAE"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for mode, group in summary.groupby("mode"):
            group = group.sort_values("seen")
            ax.plot(group["seen"], group[f"seen_{metric}"], "o-",
                    label=f"{mode}: seen")
            if group[f"unseen_{metric}"].notna().any():
                ax.plot(group["seen"], group[f"unseen_{metric}"], "s--",
                        label=f"{mode}: unseen")
        ax.set_xlabel("Seen fraction")
        ax.set_ylabel(metric)
        ax.set_title(f"California Housing: seen vs unseen {metric}")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"ca_seen_unseen_{metric.lower()}.png"), dpi=300)
        plt.close(fig)

def _aggregate_step(df, col):
    g = df.groupby("step")[col].agg(["mean", "std"]).reset_index()
    g["std"] = g["std"].fillna(0.0)
    return g["step"].values, g["mean"].values, g["std"].values

def plot_california_panels(df, seen, mode, fig_dir):

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    specs = [
        (axes[0, 0], "theta_error", r"$E_\theta$", "Parameter error"),
        (axes[0, 1], "E_M",         r"$E_M$",     "Matrix statistic error"),
        (axes[1, 0], "E_B",         r"$E_B$",     "Vector statistic error"),
        (axes[1, 1], "mass_drift",  "Mass drift", "Invariant conservation"),
    ]

    for ax, col, ylab, title in specs:
        if col not in df.columns:
            continue
        s, m, sd = _aggregate_step(df, col)
        # floor for log scale
        floor = 1e-16 if col == "mass_drift" else 1e-30
        m = np.maximum(m, floor)
        ax.plot(s, m, label="mean")
        ax.fill_between(s, np.maximum(m - sd, floor), m + sd, alpha=0.2)
        ax.set_yscale("log")
        ax.set_xlabel("Gossip step")
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)

    plt.suptitle(f"California Housing — Seen={seen:.2f}, Mode={mode}")
    plt.tight_layout()
    fname = f"ca_convergence_Seen{int(round(seen*100)):03d}_Mode{mode}.png"
    plt.savefig(os.path.join(fig_dir, fname), dpi=300)
    plt.close()


def plot_ca_mode_comparison(conv_all, seen, fig_dir):
    plt.figure(figsize=(9, 6))
    for mode, group in conv_all[conv_all["seen"] == seen].groupby("mode"):
        s, m, sd = _aggregate_step(group, "theta_error")
        mask = m > 1e-14
        plt.plot(s[mask], m[mask], label=f"{mode}", linewidth=1.7)
        plt.fill_between(s[mask],
                         np.maximum(m[mask] - sd[mask], 1e-14),
                         m[mask] + sd[mask], alpha=0.2)
    plt.yscale("log")
    plt.xlabel("Gossip step")
    plt.ylabel(r"$E_\theta$")
    plt.title(f"Full vs coordinate gossip — Seen={seen:.2f}")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir,
                f"ca_mode_comparison_Seen{int(round(seen*100)):03d}.png"),
                dpi=300)
    plt.close()


def plot_ca_seen_comparison(conv_all, mode, fig_dir):
    """
    theta error vs step for different seen_fractions at a fixed gossip_mode
    """
    plt.figure(figsize=(9, 6))
    for seen, group in conv_all[conv_all["mode"] == mode].groupby("seen"):
        s, m, sd = _aggregate_step(group, "theta_error")
        mask = m > 1e-14
        plt.plot(s[mask], m[mask],
                 label=f"Seen={seen:.2f}", linewidth=1.7)
        plt.fill_between(s[mask],
                         np.maximum(m[mask] - sd[mask], 1e-14),
                         m[mask] + sd[mask], alpha=0.15)
    plt.yscale("log")
    plt.xlabel("Gossip step")
    plt.ylabel(r"$E_\theta$")
    plt.title(f"Effect of seen fraction — mode={mode}")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"ca_seen_comparison_Mode{mode}.png"),
                dpi=300)
    plt.close()


def plot_ca_communication_tradeoff(conv_all, p, fig_dir):
    """
    theta error vs total scalar values transmitted, full-state gossip only
    """
    plt.figure(figsize=(9, 6))
    full = conv_all[conv_all["mode"] == "full"]
    for seen, group in full.groupby("seen"):
        m = p * (p + 1) // 2 + p
        cost = 2 * m
        s, m, sd = _aggregate_step(group, "theta_error")
        x = s * cost
        plt.plot(x, np.where(m > 0, m, np.nan),
                 label=f"Seen={seen:.2f} (cost/step={cost})",
                 linewidth=1.6)
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Total scalar values transmitted")
    plt.ylabel(r"$E_\theta$")
    plt.title("Communication trade-off on California Housing (full-state gossip)")
    plt.grid(True, which="both", linestyle="--", alpha=0.6)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "ca_communication_tradeoff.png"),
                dpi=300)
    plt.close()


def tolerance_crossing_table(conv_all, tolerances, p, fig_dir):
    rows = []
    for (seen, mode, seed), g in conv_all.groupby(["seen", "mode", "seed"]):
        g = g.sort_values("step")
        m = p * (p + 1) // 2 + p
        cost_per_step = 2 * m if mode == "full" else 2
        for tol in tolerances:
            hit = g[g["theta_error"] <= tol]
            if len(hit) == 0:
                rows.append({
                    "seen": seen, "mode": mode, "seed": seed,
                    "tolerance": tol,
                    "steps_to_tol": np.nan,
                    "scalars_transmitted": np.nan,
                    "reached": False,
                })
            else:
                steps_to_tol = int(hit["step"].iloc[0])
                rows.append({
                    "seen": seen, "mode": mode, "seed": seed,
                    "tolerance": tol,
                    "steps_to_tol": steps_to_tol,
                    "scalars_transmitted": steps_to_tol * cost_per_step,
                    "reached": True,
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(fig_dir, "ca_tolerance_crossing.csv"), index=False)
    agg = df.groupby(["seen", "mode", "tolerance"]).agg(
        reached_frac=("reached", "mean"),
        steps_mean=("steps_to_tol", "mean"),
        steps_std=("steps_to_tol", "std"),
        scalars_mean=("scalars_transmitted", "mean"),
        scalars_std=("scalars_transmitted", "std"),
    ).reset_index()
    agg.to_csv(os.path.join(fig_dir, "ca_tolerance_crossing_summary.csv"), index=False)
    print("\nTolerance crossing summary:")
    print(agg.to_string(index=False))
    return df


def parse_folder(name):

    parts = name.split("_")
    # parts[0] = 'CA'
    # parts[1] = 'Seen050'
    # parts[2] = 'Modefull' or 'Modecoord'
    # parts[3] = 'seed1'
    seen = int(parts[1][4:]) / 100.0
    mode = parts[2][4:]
    seed = int(parts[3][4:])
    return seen, mode, seed


# Data loading
def load_runs(base_dir):
    run_folders = sorted(glob.glob(os.path.join(base_dir, "CA_Seen*_Mode*_seed*")))
    if not run_folders:
        print(f"No CA_* folders found in {base_dir}")
        return None, None

    conv_frames = []
    metrics_rows = []

    for folder in run_folders:
        name = os.path.basename(folder)
        seen, mode, seed = parse_folder(name)

        conv_path = os.path.join(folder, "convergence.csv")
        if not os.path.exists(conv_path):
            print(f"Warning: {conv_path} not found, skipping.")
            continue

        df = pd.read_csv(conv_path)
        if "scalars_transmitted" in df:
            cost = 108 if mode == "full" else 2
            if not df["scalars_transmitted"].eq(df["step"] * cost).all():
                raise ValueError(f"Communication totals disagree with the raw steps in {conv_path}")
        df["seen"] = seen
        df["mode"] = mode
        df["seed"] = seed
        conv_frames.append(df)

        met_path = os.path.join(folder, "metrics.json")
        if os.path.exists(met_path):
            with open(met_path) as f:
                met = json.load(f)
            row = {
                "seen": seen,
                "mode": mode,
                "seed": seed,
                "rel_error_seen":   met.get("rel_error_seen"),
                "rel_error_global": met.get("rel_error_global"),
                "final_stable_fraction": met.get("final_stable_fraction"),
            }
            seen_block = met.get("Seen") or {}
            row["seen_RMSE"] = seen_block.get("RMSE")
            row["seen_MAE"]  = seen_block.get("MAE")
            row["seen_R2"]   = seen_block.get("R2")
            unseen_block = met.get("Unseen") or {}
            row["unseen_RMSE"] = unseen_block.get("RMSE")
            row["unseen_MAE"]  = unseen_block.get("MAE")
            row["unseen_R2"]   = unseen_block.get("R2")
            if "centralized" in met:
                for subset in ("Seen", "Unseen"):
                    baseline = met["centralized"][subset] or {}
                    for metric in ("RMSE", "R2"):
                        row[f"ols_{subset.lower()}_{metric}"] = baseline.get(metric)
                row["n_unseen"] = met["n_unseen"]
            metrics_rows.append(row)

    if not conv_frames:
        print("No convergence.csv files loaded.")
        return None, None

    conv_all = pd.concat(conv_frames, ignore_index=True)
    metrics_df = pd.DataFrame(metrics_rows) if metrics_rows else pd.DataFrame()
    for col in metrics_df.columns:
        if col.startswith(("seen_", "unseen_", "ols_")):
            metrics_df[col] = pd.to_numeric(metrics_df[col], errors="coerce")
    return conv_all, metrics_df


# slope fitting helper
def write_prediction_comparison(metrics_df, fig_dir):
    """Pair OLS and gossip on the same endpoint-eligible seeds."""
    if "n_unseen" not in metrics_df or metrics_df["n_unseen"].isna().any():
        print("Paired prediction table requires the centralized scores from the updated runner.")
        return
    rows = []
    for (seen, mode), group in metrics_df.groupby(["seen", "mode"]):
        for split in ("seen", "unseen"):
            if split == "unseen" and group["n_unseen"].eq(0).all():
                continue
            eligible = group[f"{split}_RMSE"].notna()
            matched = group.loc[eligible]
            row = dict(seen=seen, mode=mode, subset=split,
                       seeds_with_predictor=int(eligible.sum()), seeds_run=len(group),
                       usable_percent=100 * group["final_stable_fraction"].mean())
            for metric in ("RMSE", "R2"):
                row[f"OLS_{metric}"] = matched[f"ols_{split}_{metric}"].mean()
                row[f"gossip_{metric}"] = matched[f"{split}_{metric}"].mean()
            rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(os.path.join(fig_dir, "ca_paired_predictions.csv"), index=False)
    metrics_df.to_csv(os.path.join(fig_dir, "ca_predictions_by_seed.csv"), index=False)
    with open(os.path.join(fig_dir, "ca_prediction_table.tex"), "w", encoding="utf-8") as out:
        print(r"\begin{table}[htbp]\centering\small", file=out)
        print(r"\caption{California Housing predictions at the final checkpoint. RMSE is in units of \$100,000. Scores are means over the same seeds for both methods. Runs gives the number of seeds yielding a gossip predictor out of all seeds run. Usable is the mean percentage of usable agents across all seeds.}", file=out)
        print(r"\label{tab:ca-paired-predictions}\begin{tabular}{lllrrrrrr}\hline", file=out)
        print(r"Seen & Mode & Subset & Runs & Usable (\%) & \multicolumn{2}{c}{RMSE} & \multicolumn{2}{c}{$R^2$} \\", file=out)
        print(r" & & & & & OLS & Gossip & OLS & Gossip \\ \hline", file=out)
        for row in table.itertuples(index=False):
            fields = [f"{row.seen:.2f}", row.mode, row.subset,
                      f"{row.seeds_with_predictor}/{row.seeds_run}", f"{row.usable_percent:.1f}"]
            for value in (row.OLS_RMSE, row.gossip_RMSE, row.OLS_R2, row.gossip_R2):
                fields.append(f"{value:.4f}" if pd.notna(value) else "--")
            print(" & ".join(fields) + r" \\", file=out)
        print(r"\hline\end{tabular}\end{table}", file=out)


def fit_slopes(conv_all):
    """
    log(theta_error)
    """
    rows = []
    for (seen, mode, seed), g in conv_all.groupby(["seen", "mode", "seed"]):
        g = g[(g["theta_error"] > 1e-12) & (g["theta_error"] < 0.5)]
        if len(g) < 10:
            continue
        x = g["step"].values
        y = np.log(g["theta_error"].values)
        slope, intercept = np.polyfit(x, y, 1)
        rows.append({
            "seen": seen, "mode": mode, "seed": seed,
            "slope_per_step": slope, "intercept": intercept,
        })
    if not rows:
        return pd.DataFrame()
    slope_df = pd.DataFrame(rows)
    summary = slope_df.groupby(["seen", "mode"])["slope_per_step"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    summary.rename(columns={
        "mean": "slope_mean", "std": "slope_std", "count": "n"
    }, inplace=True)
    return summary


def write_final_stability_report(records, output_dir, group_columns):
    from pathlib import Path

    if not records:
        print("No final-stability records available.")
        return
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = pd.DataFrame(records)
    f = pd.to_numeric(runs["final_stable_fraction"], errors="coerce")
    known = np.isfinite(f) & f.between(0, 1)
    runs["stability_status"] = np.select(
        [known & f.eq(1.0), known & f.lt(1.0)],
        ["full_100_percent", "below_100_percent"], default="unknown")
    runs["final_stable_fraction"] = f.where(known)
    theta = pd.to_numeric(runs["final_Etheta"], errors="coerce")
    runs["final_Etheta"] = theta.where(np.isfinite(theta))
    runs = runs.sort_values(group_columns + ["seed"] if "seed" in runs else group_columns)
    runs_path = output_dir / "final_stability_runs.csv"
    runs.to_csv(runs_path, index=False)

    rows = []
    for group, data in runs.groupby(group_columns, dropna=False, sort=True):
        if not isinstance(group, tuple):
            group = (group,)
        status = data["stability_status"]
        full = status.eq("full_100_percent")
        below = status.eq("below_100_percent")
        known_count = int((full | below).sum())
        row = dict(zip(group_columns, group))
        row.update(
            n_runs=len(data),
            n_full_100_percent=int(full.sum()),
            n_below_100_percent=int(below.sum()),
            n_unknown=int(status.eq("unknown").sum()),
            fraction_full_among_known=(float(full.sum()) / known_count
                                       if known_count else np.nan),
            mean_final_stable_fraction=float(data["final_stable_fraction"].mean()),
            mean_final_Etheta_full=float(data.loc[full, "final_Etheta"].mean()),
            mean_final_Etheta_below=float(data.loc[below, "final_Etheta"].mean()),
            n_finite_Etheta_full=int(data.loc[full, "final_Etheta"].notna().sum()),
            n_finite_Etheta_below=int(data.loc[below, "final_Etheta"].notna().sum()),
        )
        rows.append(row)
    summary = pd.DataFrame(rows)
    summary_path = output_dir / "final_stability_summary.csv"
    summary.to_csv(summary_path, index=False)
    counts = runs["stability_status"].value_counts()
    print("\nFinal numerical usability (100% means every agent is eligible):")
    print(f"  100%: {counts.get('full_100_percent', 0)} runs; "
          f"below 100%: {counts.get('below_100_percent', 0)} runs; "
          f"unknown: {counts.get('unknown', 0)} runs")
    print(summary.to_string(index=False))
    print(f"Run-by-run classification: {runs_path}")
    print(f"Configuration breakdown: {summary_path}")
    print("Existing E_theta plots and aggregates are unchanged; this report "
          "does not exclude partially usable runs.")

def main():
    # find base directory
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        exp_dir = os.path.dirname(os.path.abspath(__file__))
        candidates = sorted(glob.glob(os.path.join(exp_dir, "results", "exp_H_*")))
        if not candidates:
            print("No exp_H_* folders found under experiments/exp_H/results/.")
            return
        base_dir = candidates[-1]

    print(f"Analysing results in: {base_dir}")
    fig_dir = os.path.join(base_dir, "analysis_figures")
    os.makedirs(fig_dir, exist_ok=True)

    conv_all, metrics_df = load_runs(base_dir)
    if conv_all is None:
        return
    stability_records = []
    metrics_by_run = {}
    if not metrics_df.empty:
        metrics_by_run = {
            (float(r.seen), str(r.mode), int(r.seed)): r.final_stable_fraction
            for r in metrics_df.itertuples(index=False)
        }
    for (seen, mode, seed), group in conv_all.groupby(["seen", "mode", "seed"]):
        last = group.sort_values("step").iloc[-1]
        key = (float(seen), str(mode), int(seed))
        fraction = last.get("stable_fraction", np.nan)
        source = "convergence.csv" if "stable_fraction" in group else "missing"
        if key in metrics_by_run and pd.notna(metrics_by_run[key]):
            fraction = metrics_by_run[key]
            source = "metrics.json"
        stability_records.append({
            "run": f"CA_Seen{int(round(seen*100)):03d}_Mode{mode}_seed{seed}",
            "seen": seen, "mode": mode, "seed": seed,
            "last_logged_step": last["step"],
            "final_stable_fraction": fraction,
            "stability_source": source,
            "final_Etheta": last.get("theta_error", np.nan),
        })
    write_final_stability_report(stability_records, fig_dir, ["seen", "mode"])

    p = 9
    print(f"Loaded {conv_all['seed'].nunique()} seeds, "
          f"{conv_all.groupby(['seen','mode']).ngroups} configurations.")

    # convergence panels
    for (seen, mode), g in conv_all.groupby(["seen", "mode"]):
        plot_california_panels(g, seen, mode, fig_dir)
        print(f"  panel: Seen={seen:.2f}, mode={mode}")

    # full vs coordinate
    for seen in sorted(conv_all["seen"].unique()):
        plot_ca_mode_comparison(conv_all, seen, fig_dir)


    for mode in sorted(conv_all["mode"].unique()):
        plot_ca_seen_comparison(conv_all, mode, fig_dir)

    plot_ca_communication_tradeoff(conv_all, p, fig_dir)

    if not metrics_df.empty:
        write_prediction_comparison(metrics_df, fig_dir)
        plot_seen_vs_unseen_metrics(metrics_df, fig_dir)

        agg = metrics_df.groupby(["seen", "mode"]).agg(
            seen_RMSE_mean=("seen_RMSE", "mean"),
            seen_RMSE_std=("seen_RMSE", "std"),
            seen_MAE_mean=("seen_MAE", "mean"),
            seen_R2_mean=("seen_R2", "mean"),
            unseen_RMSE_mean=("unseen_RMSE", "mean"),
            unseen_RMSE_std=("unseen_RMSE", "std"),
            unseen_MAE_mean=("unseen_MAE", "mean"),
            unseen_R2_mean=("unseen_R2", "mean"),
            rel_error_seen_mean=("rel_error_seen", "mean"),
            rel_error_global_mean=("rel_error_global", "mean"),
            final_stable_fraction_mean=("final_stable_fraction", "mean"),
        ).reset_index()
        agg.to_csv(os.path.join(fig_dir, "ca_metrics_summary.csv"), index=False)
        print("\nMetrics summary:")
        print(agg.to_string(index=False))
        print(f"\nSaved to {os.path.join(fig_dir, 'ca_metrics_summary.csv')}")

    tolerances = [1e-1, 5e-2, 1e-2, 5e-3, 1e-3]
    tolerance_crossing_table(conv_all, tolerances, p, fig_dir)

    slope_summary = fit_slopes(conv_all)
    if not slope_summary.empty:
        slope_summary.to_csv(os.path.join(fig_dir, "ca_slope_summary.csv"), index=False)
        print("\nPer-step log-theta-error slopes (more negative = faster):")
        print(slope_summary.to_string(index=False))

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()