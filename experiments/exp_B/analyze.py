"""Full-state gossip scaling analysis"""
from pathlib import Path
import argparse
import json
import re
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "results"
DEFAULT_RELATIVE_MSE_OFFSET = 1e-12
FOLDER_PATTERN = re.compile(r"A(?P<A>\d+)_D(?P<D>\d+)_seed(?P<seed>\d+)$")


def initial_statistics(csv_file, A, D):
    data = pd.read_csv(csv_file)
    features = [c for c in data.columns if c != "y"]
    if len(data) != A or len(features) != D or "y" not in data:
        raise ValueError(f"{csv_file}: dimensions do not match A={A}, D={D}")
    Z = np.column_stack((data[features].to_numpy(dtype=float), np.ones(A)))
    y = data["y"].to_numpy(dtype=float)
    M_agents = np.einsum("ij,ik->ijk", Z, Z)
    B_agents = Z * y[:, None]
    M_global = M_agents.mean(axis=0)
    B_global = B_agents.mean(axis=0)
    psi0 = (np.square(M_agents - M_global).sum()
            + np.square(B_agents - B_global).sum())
    if not np.isfinite(psi0) or psi0 <= 0:
        raise ValueError(f"{csv_file}: initial disagreement must be positive and finite")
    return float(psi0), float(np.square(M_global).sum()), float(np.square(B_global).sum())


def exact_reference(interactions, A):
    """Expected normalized potential under uniform full-state pair averaging."""
    return np.exp(np.asarray(interactions, dtype=float) * np.log1p(-1.0 / (A - 1)))


def locate_data_dir(root, override):
    if override is not None:
        return override.expanduser().resolve()
    for location in (root / "data", HERE / "data"):
        if location.is_dir():
            return location
    raise FileNotFoundError(
        f"Dataset CSV directory not found. Checked {root / 'data'} and "
        f"{HERE / 'data'}. Generate exp_B data first or pass --data-dir."
    )


def plot_four_panels(df, A, D, fig_dir):
    columns = ["theta_error", "E_M", "E_B", "mass_drift"]
    stats = df.groupby("step")[columns].agg(["mean", "std"]).reset_index()
    stats.columns = ["step", "theta_mean", "theta_std", "M_mean", "M_std",
                     "B_mean", "B_std", "mass_mean", "mass_std"]

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    panels = [
        (axes[0, 0], "theta", r"$E_\theta$", "Parameter error"),
        (axes[0, 1], "M", r"$E_M$", "Matrix statistic error"),
        (axes[1, 0], "B", r"$E_B$", "Vector statistic error"),
        (axes[1, 1], "mass", "Mass drift", "Invariant conservation"),
    ]
    for ax, key, ylabel, title in panels:
        mean, std = stats[f"{key}_mean"], stats[f"{key}_std"].fillna(0)
        ax.plot(stats["step"], mean)
        ax.fill_between(stats["step"], mean - std, mean + std, alpha=0.2)
        ax.set_yscale("log")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, which="both", linestyle="--", alpha=0.6)
    fig.suptitle(f"Convergence for A={A}, d={D}")
    fig.tight_layout()
    fig.savefig(fig_dir / f"convergence_A{A}_D{D}.png", dpi=300)
    plt.close(fig)



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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path,
                        help="Results folder; default: latest exp_B_* inside exp_B/results")
    parser.add_argument("--data-dir", type=Path,
                        help="Dataset CSV directory; default: archived data or exp_B/data")
    parser.add_argument("--relative-mse-offset", type=float,
                        default=DEFAULT_RELATIVE_MSE_OFFSET,
                        help="E_M/E_B denominator offset (default: 1e-12, verified in main.py)")
    args = parser.parse_args()
    if args.relative_mse_offset < 0:
        parser.error("--relative-mse-offset must be nonnegative")

    if args.results_dir is None:
        candidates = sorted(p for p in RESULTS_DIR.glob("exp_B_*") if p.is_dir())
        if not candidates:
            parser.error(f"No exp_B_* result folders found in {RESULTS_DIR}")
        root = candidates[-1]
    else:
        root = args.results_dir.expanduser().resolve()
    if not root.is_dir():
        parser.error(f"Results directory does not exist: {root}")
    data_dir = locate_data_dir(root, args.data_dir)
    print(f"Analyzing results in: {root}", flush=True)
    print(f"Loading datasets from: {data_dir}", flush=True)

    manifest = root / "experiment.json"
    config = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if config.get("gossip_mode", "full") != "full":
        raise ValueError("This analysis applies to full-state gossip only")
    budget = config.get("steps")

    folders = []
    for child in root.iterdir():
        match = FOLDER_PATTERN.fullmatch(child.name)
        if child.is_dir() and match and (child / "convergence.csv").is_file():
            folders.append((child, *(int(match.group(k)) for k in ("A", "D", "seed"))))
    if not folders:
        raise FileNotFoundError(f"No A*_D*_seed*/convergence.csv logs in {root}")
    folders.sort(key=lambda item: (item[1], item[2], item[3]))

    frames, slopes, incomplete = [], [], []
    stability_records = []
    diagnostic_frames = []
    for folder, A, D, seed in folders:
        name = folder.name
        csv_file = data_dir / f"{name}.csv"
        if not csv_file.is_file():
            raise FileNotFoundError(f"{name}: missing pre-generated dataset {csv_file}")
        df = pd.read_csv(folder / "convergence.csv").sort_values("step")
        required = {"step", "E_M", "E_B"}
        if not required.issubset(df.columns):
            raise ValueError(f"{name}: missing convergence columns {sorted(required-set(df.columns))}")
        if df.empty or df["step"].isna().any() or df["step"].duplicated().any():
            raise ValueError(f"{name}: invalid or duplicate checkpoints")
        if ((df["step"] < 0) | (df["step"] % 1 != 0)).any():
            raise ValueError(f"{name}: step must contain nonnegative integers")
        # four-panel
        diagnostic_columns = {"step", "theta_error", "E_M", "E_B", "mass_drift"}
        if diagnostic_columns.issubset(df.columns):
            diagnostic = df[list(("step", "theta_error", "E_M", "E_B", "mass_drift"))].copy()
            diagnostic["A"], diagnostic["D"], diagnostic["seed"] = A, D, seed
            diagnostic_frames.append(diagnostic)
        else:
            warnings.warn(f"{name}: four-panel inputs missing: "
                          f"{sorted(diagnostic_columns-set(df.columns))}")
        last = df.iloc[-1]
        fraction = last.get("stable_fraction", np.nan)
        fraction_source = "convergence.csv" if "stable_fraction" in df else "missing"
        met_file = folder / "metrics.json"
        final_metrics = {}
        if met_file.is_file():
            with met_file.open() as handle:
                final_metrics = json.load(handle)
            if "final_stable_fraction" in final_metrics:
                fraction = final_metrics["final_stable_fraction"]
                fraction_source = "metrics.json"
        stability_records.append({
            "run": name, "A": A, "D": D, "seed": seed,
            "last_logged_step": int(last["step"]),
            "final_stable_fraction": fraction,
            "stability_source": fraction_source,
            "final_Etheta": last.get("theta_error", np.nan),
        })
        psi0, mn, bn = initial_statistics(csv_file, A, D)
        if "Psi" in df.columns:
            psi = df["Psi"].to_numpy(dtype=float)
        else:
            em = df["E_M"].to_numpy(dtype=float)
            eb = df["E_B"].to_numpy(dtype=float)
            eps = args.relative_mse_offset
            psi = A * (em * (mn + eps) + eb * (bn + eps))
        if not np.isfinite(psi).all() or np.any(psi < -1e-12):
            raise ValueError(f"{name}: invalid reconstructed disagreement")
        ratio = np.maximum(psi, 0) / psi0
        if ratio.max() > 1 + 1e-6:
            raise ValueError(f"{name}: disagreement exceeds initial potential; verify metric definitions")
        if np.any(np.diff(ratio) > 1e-7):
            warnings.warn(f"{name}: disagreement is not monotone; inspect the source logs")

        log_steps = df["step"].to_numpy(dtype=int)

        if "interactions" in df:
            interactions = df["interactions"].to_numpy(dtype=int)
        else:
            has_zero = bool(np.any(log_steps == 0))
            if has_zero:
                at_zero = float(ratio[log_steps == 0][0])
                post_interaction_zero = not np.isclose(at_zero, 1.0, rtol=1e-7, atol=1e-10)
            else:
                post_interaction_zero = True
            interactions = log_steps + int(post_interaction_zero)
        if budget is not None and interactions[-1] < int(budget) and final_metrics.get("stop_reason") != "all_inactive":
            incomplete.append(f"{name}: last interaction={interactions[-1]}, budget={budget}")

        run = pd.DataFrame({
            "step": log_steps,
            "interactions": interactions,
            "Psi_norm": ratio,
            "A": A, "D": D, "seed": seed,
            "scaled_step": interactions / (A - 1),
            "exact": exact_reference(interactions, A),
        })

        if not np.any(interactions == 0):
            run = pd.concat([pd.DataFrame({
                "step": [-1], "interactions": [0], "Psi_norm": [1.0],
                "A": [A], "D": [D], "seed": [seed],
                "scaled_step": [0.0], "exact": [1.0],
            }), run], ignore_index=True)
        frames.append(run)
        fitting = run[(run["Psi_norm"] > 1e-6) & (run["Psi_norm"] < .5)]
        if len(fitting) >= 5:
            slope = np.polyfit(fitting["scaled_step"], np.log(fitting["Psi_norm"]), 1)[0]
            slopes.append({"A": A, "D": D, "seed": seed,
                           "slope": float(slope),
                           "exact_slope": float((A-1) * np.log1p(-1.0/(A-1)))})

    if incomplete:
        print(f"Warning: {len(incomplete)} runs end before the manifest budget.")
        for item in incomplete[:5]:
            print(f"  {item}")
        if len(incomplete) > 5:
            print(f"  ... and {len(incomplete) - 5} others")
        print("These runs are still plotted, but early stopping can invalidate the exact-theory comparison.")

    all_runs = pd.concat(frames, ignore_index=True)
    out = root / "analysis_figures"
    out.mkdir(exist_ok=True)
    write_final_stability_report(stability_records, out, ["A", "D"])
    # four-panel
    if diagnostic_frames:
        diagnostics = pd.concat(diagnostic_frames, ignore_index=True)
        configurations = sorted(set(zip(diagnostics["A"], diagnostics["D"])))
        A_max, D_max = configurations[-1]
        selection = diagnostics[(diagnostics["A"] == A_max) &
                                (diagnostics["D"] == D_max)]
        print(f"Generating four-panel convergence plot (A={A_max}, D={D_max})...", flush=True)
        plot_four_panels(selection, A_max, D_max, out)
        print(f"Saved {out / f'convergence_A{A_max}_D{D_max}.png'}", flush=True)
    else:
        warnings.warn("No four-panel diagnostics available: source logs lack "
                      "theta_error and/or mass_drift")
    print("Generating normalized disagreement collapse plot...", flush=True)
    all_runs.to_csv(out / "Psi_normalized_runs.csv", index=False)
    summaries = []
    A_values = sorted(all_runs["A"].unique())
    D_values = sorted(all_runs["D"].unique())
    fig, axes = plt.subplots(1, len(D_values),
                            figsize=(max(6, 5*len(D_values)), 4.8),
                            sharey=True, squeeze=False)
    colors = plt.get_cmap("tab10")
    for ax, D in zip(axes[0], D_values):
        for color_index, A in enumerate(A_values):
            group = all_runs[(all_runs["A"] == A) & (all_runs["D"] == D)]
            if group.empty:
                continue
            summary = group.groupby("interactions")["Psi_norm"].agg(
                ["mean", "std", "count"]).reset_index()
            summary["A"], summary["D"] = A, D
            summary["scaled_step"] = summary["interactions"] / (A-1)
            summary["exact"] = exact_reference(summary["interactions"], A)
            summaries.append(summary)
            visible = summary["mean"] > 1e-14
            x = summary.loc[visible, "scaled_step"].to_numpy()
            means = summary.loc[visible, "mean"].to_numpy()
            std = summary.loc[visible, "std"].fillna(0).to_numpy()
            color = colors(color_index % 10)
            n = int(group["seed"].nunique())
            ax.plot(x, means, color=color, label=f"A={A}, mean (n={n})")
            ax.fill_between(x, np.maximum(means - std, 1e-14), means + std,
                            color=color, alpha=0.13)
            ax.plot(x, summary.loc[visible, "exact"], color=color,
                    linestyle="--", label=f"A={A}, uniform-pair theory")
        ax.set(title=f"d={D}", xlabel=r"Scaled interactions $t/(A-1)$",
               yscale="log", ylim=(1e-14, 2))
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel(r"$\Psi(t)/\Psi(0)$")
    fig.suptitle("Full-state gossip: normalized disagreement and exact uniform-pair expectation")
    fig.tight_layout()
    fig.savefig(out / "Phi_norm_collapse.png", dpi=300)
    plt.close(fig)
    pd.concat(summaries, ignore_index=True).to_csv(out / "Psi_mean_summary.csv", index=False)
    print("Fitting slopes...", flush=True)
    if slopes:
        fits = pd.DataFrame(slopes)
        fits.to_csv(out / "Psi_slope_runs.csv", index=False)
        slope_summary = fits.groupby(["A", "D"]).agg(
            slope_mean=("slope", "mean"),
            slope_std=("slope", "std"),
            n=("slope", "count"),
            exact_slope=("exact_slope", "first"),
        )
        slope_summary.to_csv(out / "Phi_slope_summary.csv")
        print("Normalized disagreement slopes by configuration:")
        print(slope_summary.to_string())
        print(f"Overall: mean={fits['slope'].mean():.3f}, "
              f"std={fits['slope'].std():.3f}, n={len(fits)}")
    else:
        print("No slope fits could be performed.")
    print(f"Saved results to {out}; {len(frames)} runs analyzed.")
    print("Shaded bands are standard deviations across runs, not confidence intervals.")
    print("Note: theoretical curves assume uniformly sampled pairs without node deactivation.")


if __name__ == "__main__":
    main()
