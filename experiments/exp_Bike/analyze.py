from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
NAME_RE = re.compile(r"BS_Seen(\d+)_Mode(full|coord)_seed(\d+)$")


def parse_name(name):
    m = NAME_RE.fullmatch(name)
    if not m:
        raise ValueError(f"Invalid run directory: {name}")
    return int(m.group(1)) / 100, m.group(2), int(m.group(3))


def load_runs(root):
    traces, metrics, final_agents = [], [], []
    for folder in sorted(root.glob("BS_Seen*_Mode*_seed*")):
        if not folder.is_dir() or not NAME_RE.fullmatch(folder.name):
            continue
        seen, mode, seed = parse_name(folder.name)
        conv_file = folder / "convergence.csv"
        if not conv_file.is_file():
            print(f"Skipping missing convergence file: {conv_file}")
            continue
        df = pd.read_csv(conv_file)
        if df.empty or "step" not in df or "theta_error" not in df:
            print(f"Skipping empty/incomplete convergence file: {conv_file}")
            continue
        for column, value in (("seen", seen), ("mode", mode), ("seed", seed)):
            df[column] = value
        df["run"] = folder.name
        traces.append(df)
        met_path = folder / "metrics.json"
        met = json.loads(met_path.read_text()) if met_path.is_file() else {}
        latest = df.sort_values("step").iloc[-1]
        stable = met.get("final_stable_fraction", latest.get("stable_fraction", np.nan))
        try:
            stable = float(stable)
        except (ValueError, TypeError):
            stable = np.nan
        if np.isfinite(stable) and np.isclose(stable, 1.0, rtol=0, atol=1e-12):
            category = "full_100_percent"
        elif np.isfinite(stable) and 0 <= stable < 1:
            category = "below_100_percent"
        else:
            category = "unknown"
        row = {"run": folder.name, "seen": seen, "mode": mode, "seed": seed,
               "final_stable_fraction": stable, "stability_category": category,
               "final_theta_error": latest["theta_error"],
               "steps_logged": int(latest["step"]),
               "metrics_present": bool(met)}
        for col in ("rel_error_seen", "rel_error_global", "n_agents", "n_stable",
                    "n_finite_eligible", "objective_usable_fraction"):
            row[col] = met.get(col, np.nan)
        for part in ("Seen", "Unseen"):
            sub = met.get(part) or {}
            for metric in ("RMSE", "MAE", "R2"):
                row[f"{part.lower()}_{metric}"] = sub.get(metric, np.nan)
        for tag in ("seen", "all"):
            for col in (f"central_min_{tag}", f"agent_sse_mean_{tag}",
                        f"objective_gap_mean_{tag}", f"objective_ratio_mean_{tag}"):
                row[col] = met.get(col, latest.get(col, np.nan))
        metrics.append(row)
        agent_file = folder / "final_agent_objectives.csv"
        if agent_file.is_file():
            agents = pd.read_csv(agent_file)
            for field, value in (("seen", seen), ("mode", mode), ("seed", seed)):
                agents[field] = value
            agents["run"] = folder.name
            final_agents.append(agents)
    if not traces:
        raise FileNotFoundError(f"No valid Bike Sharing convergence files in {root}")
    return pd.concat(traces, ignore_index=True), pd.DataFrame(metrics), final_agents


def by_step(df, field):
    if field not in df:
        return None
    group = df.groupby("step", as_index=False)[field].agg(["mean", "std", "count"]).reset_index()
    return (group["step"].to_numpy(float),
            group["mean"].to_numpy(float),
            group["std"].fillna(0).to_numpy(float))


def curve(ax, df, field, label, log=True, band=True):
    result = by_step(df, field)
    if result is None:
        return
    steps, means, devs = result
    good = np.isfinite(steps) & np.isfinite(means)
    if log:
        good &= means > 0
    if not good.any():
        return
    (line,) = ax.plot(steps[good], means[good], label=label, linewidth=1.6)
    if band:
        low = means[good] - devs[good]
        if log:
            low = np.maximum(low, 1e-30)
        ax.fill_between(steps[good], low, means[good] + devs[good],
                        color=line.get_color(), alpha=.15)
    ax.set_xscale("log")
    if log:
        ax.set_yscale("log")
    ax.grid(alpha=.3)


def make_panels(traces, out):
    specifications = (("theta_error", r"$E_\theta$"),
                      ("E_M", r"$E_M$"),
                      ("E_B", r"$E_B$"),
                      ("mass_drift", "Mass drift"))
    for (seen, mode), df in traces.groupby(["seen", "mode"]):
        fig, axs = plt.subplots(2, 2, figsize=(11, 8))
        for ax, (field, title) in zip(axs.flat, specifications):
            curve(ax, df, field, "Mean ± SD")
            ax.set_title(title)
            ax.set_xlabel("Gossip interaction")
            if ax.get_legend_handles_labels()[0]:
                ax.legend()
        fig.suptitle(f"Bike Sharing — seen={seen:.2f}, mode={mode}")
        fig.tight_layout()
        fig.savefig(out / f"bs_convergence_Seen{int(round(seen*100)):03d}_Mode{mode}.png", dpi=250)
        plt.close(fig)


def make_mode_seen_comparisons(traces, out):
    for seen, g in traces.groupby("seen"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for mode, df in g.groupby("mode"):
            curve(ax, df, "theta_error", f"{mode}")
        ax.set(title=f"Full vs coordinate gossip — seen={seen:.2f}",
               xlabel="Interactions", ylabel=r"$E_\theta$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"bs_mode_comparison_Seen{int(round(seen*100)):03d}.png", dpi=250)
        plt.close(fig)
    for mode, g in traces.groupby("mode"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for seen, df in g.groupby("seen"):
            curve(ax, df, "theta_error", f"seen={seen:.2f}")
        ax.set(title=f"Seen-fraction comparison — mode={mode}",
               xlabel="Interactions", ylabel=r"$E_\theta$")
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"bs_seen_comparison_Mode{mode}.png", dpi=250)
        plt.close(fig)


def communication_cost(mode, p):
    return 2 * (p * (p + 1) // 2 + p) if mode == "full" else 2


def make_communication_plot(traces, p, out):
    fig, ax = plt.subplots(figsize=(9, 5))
    for (seen, mode), g in traces.groupby(["seen", "mode"]):
        s = by_step(g, "theta_error")
        if s is None:
            continue
        steps, means, _ = s
        good = np.isfinite(means) & (means > 0) & (steps > 0)
        ax.plot(steps[good] * communication_cost(mode, p), means[good],
                label=f"seen={seen:.2f}, {mode}")
    ax.set(xscale="log", yscale="log", xlabel="Bidirectional scalar transmissions",
           ylabel=r"$E_\theta$", title="Communication vs convergence")
    ax.grid(alpha=.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "bs_communication_tradeoff.png", dpi=250)
    plt.close(fig)


def make_prediction_plot(metrics, out):
    if not {"seen_RMSE", "unseen_RMSE"}.issubset(metrics.columns):
        return
    averages = metrics.groupby(["seen", "mode"], as_index=False)[
        ["seen_RMSE", "unseen_RMSE", "seen_MAE", "unseen_MAE"]].mean()
    for measure in ("RMSE", "MAE"):
        fig, ax = plt.subplots(figsize=(9, 5))
        for mode, g in averages.groupby("mode"):
            g = g.sort_values("seen")
            ax.plot(g["seen"], g[f"seen_{measure}"], "o-", label=f"{mode}: seen")
            if g[f"unseen_{measure}"].notna().any():
                ax.plot(g["seen"], g[f"unseen_{measure}"], "s--", label=f"{mode}: unseen")
        ax.set(xlabel="Seen fraction", ylabel=measure,
               title=f"Bike Sharing: seen vs unseen {measure}")
        ax.grid(alpha=.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / f"bs_prediction_{measure.lower()}.png", dpi=250)
        plt.close(fig)


def make_objective_plots(traces, metrics, out):
    needed = ["central_min_seen", "agent_sse_mean_seen", "central_min_all", "agent_sse_mean_all"]
    if not set(needed).issubset(traces):
        print("No objective columns found; skipping objective plots (older runs).")
        return
    for tag in ("seen", "all"):
        fig, axs = plt.subplots(1, 2, figsize=(12, 4.8))
        for (seen, mode), g in traces.groupby(["seen", "mode"]):
            label = f"seen={seen:.2f}, {mode}"
            matched = g.copy()
            has_agents = matched[f"agent_sse_mean_{tag}"].notna()
            matched.loc[~has_agents, f"central_min_{tag}"] = np.nan
            fields = [f"agent_sse_mean_{tag}", f"central_min_{tag}", f"objective_gap_mean_{tag}"]
            means = matched.groupby("step")[fields].mean()
            means["contributing_seeds"] = matched.groupby("step")[fields[0]].count()
            means["mean_usable_fraction"] = matched.groupby("step")["objective_usable_fraction"].mean()
            means.to_csv(out / f"bs_plotted_{tag}_{seen:.2f}_{mode}.csv")
            for field, ax, prefix, style in zip(fields, (axs[0], axs[0], axs[1]),
                                                ("Agents ", "OLS ", ""), ("-", "--", "-")):
                series = means[field]
                ax.plot(means.index, series.where(series.gt(0)), linestyle=style,
                        label=prefix + label)
        axs[0].set(title=f"||Z_{tag} θᵢ-y||² vs centralized minimum", ylabel="Squared residual")
        axs[1].set(title=f"Excess squared residual ({tag})", ylabel="Mean excess above minimum")
        for ax in axs:
            ax.set(xscale="log", yscale="log", xlabel="Gossip interactions")
            ax.grid(alpha=.3)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / f"bs_objective_comparison_{tag}.png", dpi=250)
        plt.close(fig)
    keys = ["seen", "mode", "seed", "run", "final_stable_fraction", "objective_usable_fraction",
            "final_theta_error", "central_min_seen", "agent_sse_mean_seen",
            "objective_gap_mean_seen", "objective_ratio_mean_seen",
            "central_min_all", "agent_sse_mean_all", "objective_gap_mean_all",
            "objective_ratio_mean_all"]
    metrics[[col for col in keys if col in metrics]].to_csv(out / "bs_objective_run_comparison.csv", index=False)
    metrics.groupby(["seen", "mode"], as_index=False)[
        [col for col in keys[4:] if col in metrics and col != "final_theta_error"]
    ].agg(["mean", "std", "count"]).to_csv(out / "bs_objective_summary.csv")


def make_stability_tables(metrics, out):
    runs = metrics[["run", "seen", "mode", "seed", "final_stable_fraction",
                    "stability_category", "final_theta_error", "steps_logged",
                    "objective_usable_fraction"]].copy()
    runs.to_csv(out / "final_stability_runs.csv", index=False)
    summary_rows = []
    for (seen, mode), group in runs.groupby(["seen", "mode"]):
        for status in ("full_100_percent", "below_100_percent", "unknown"):
            sub = group[group.stability_category == status]
            vals = pd.to_numeric(sub.final_theta_error, errors="coerce")
            vals = vals[np.isfinite(vals)]
            summary_rows.append({
                "seen": seen, "mode": mode, "status": status,
                "count": len(sub), "total_runs": len(group),
                "fraction_of_runs": len(sub) / len(group),
                "mean_final_Etheta": float(vals.mean()) if len(vals) else np.nan,
                "std_final_Etheta": float(vals.std(ddof=1)) if len(vals) > 1 else np.nan,
                "mean_final_usable_fraction": sub.final_stable_fraction.mean(),
            })
    pd.DataFrame(summary_rows).to_csv(out / "final_stability_summary.csv", index=False)
    print("\nFinal numerical eligibility by configuration:")
    print(pd.DataFrame(summary_rows).to_string(index=False))


def tolerance_crossing_table(traces, p, out):
    rows = []
    for (seen, mode, seed), g in traces.groupby(["seen", "mode", "seed"]):
        g = g.sort_values("step")
        cost = communication_cost(mode, p)
        for tol in (1e-1, 5e-2, 1e-2, 5e-3, 1e-3):
            hits = g[g["theta_error"] <= tol]
            step = int(hits.step.iloc[0]) if len(hits) else np.nan
            rows.append({"seen": seen, "mode": mode, "seed": seed,
                         "tolerance": tol, "steps_to_tol": step,
                         "scalars_transmitted": step * cost if np.isfinite(step) else np.nan,
                         "reached": bool(len(hits))})
    raw = pd.DataFrame(rows)
    raw.to_csv(out / "bs_tolerance_crossing.csv", index=False)
    raw.groupby(["seen", "mode", "tolerance"], as_index=False).agg(
        reached_frac=("reached", "mean"), steps_mean=("steps_to_tol", "mean"),
        steps_std=("steps_to_tol", "std"),
        scalars_mean=("scalars_transmitted", "mean"),
        scalars_std=("scalars_transmitted", "std")
    ).to_csv(out / "bs_tolerance_crossing_summary.csv", index=False)


def slope_fits(traces, out):
    rows = []
    for (seen, mode, seed), g in traces.groupby(["seen", "mode", "seed"]):
        g = g[(g["theta_error"] > 1e-12) & (g["theta_error"] < 0.5)]
        if len(g) < 10:
            continue
        slope, intercept = np.polyfit(g.step, np.log(g.theta_error), 1)
        rows.append({"seen": seen, "mode": mode, "seed": seed,
                     "slope_per_step": slope, "intercept": intercept})
    if rows:
        summary = pd.DataFrame(rows).groupby(["seen", "mode"], as_index=False).agg(
            slope_mean=("slope_per_step", "mean"),
            slope_std=("slope_per_step", "std"), n=("slope_per_step", "count"))
        summary.to_csv(out / "bs_slope_summary.csv", index=False)


def design_diagnostic(Z, label):
    s = np.linalg.svd(Z, compute_uv=False)
    tol = np.finfo(Z.dtype).eps * max(Z.shape) * s[0]
    rank = int(np.sum(s > tol))
    maximum = float(s[0]) if len(s) else 0.0
    minimum = float(s[-1]) if len(s) else 0.0
    robust = float(s[rank-1]) if rank else 0
    return {"matrix": label, "n_rows": Z.shape[0], "n_cols": Z.shape[1],
            "rank": rank, "rank_tolerance": tol, "s_max": maximum, "s_min": minimum,
            "cond_full": maximum / minimum if minimum else np.inf,
            "cond_robust": maximum / robust if robust else np.inf,
            "s_values": json.dumps(s.tolist())}


def analyze_diagnostics(root, config, metrics, out):
    destination = root / "diagnostics"
    destination.mkdir(exist_ok=True)
    data_dir = Path(config.get("data_dir") or HERE / "data" / config.get("granularity", "hour"))
    if (root / "designs").is_dir():
        data_dir = root / "designs"
    if not data_dir.is_dir():
        data_dir = HERE / "data" / config.get("granularity", "hour")
    manifest_file = data_dir / "features.json"
    manifest = json.loads(manifest_file.read_text()) if manifest_file.exists() else {}
    relations = []
    design_rows, ols_rows = [], []
    for seen, seed in sorted(set(zip(metrics["seen"], metrics["seed"]))):
        filename = f"BS_Seen{int(round(seen*100)):03d}_seed{seed}.csv"
        datafile = data_dir / filename
        if not datafile.is_file():
            print(f"Warning: can't find {filename}; skipping archived-data diagnostics.")
            continue
        frame = pd.read_csv(datafile)
        features = sorted((c for c in frame if c.startswith("x") and c[1:].isdigit()),
                          key=lambda c: int(c[1:]))
        Z = np.column_stack((frame[features].to_numpy(dtype=float), np.ones(len(frame))))
        y = frame["y"].to_numpy(dtype=float)
        mask = frame["split"].eq("seen").to_numpy()
        for name, subset in (("X_seen_aug", mask), ("X_all_aug", np.ones(len(y), dtype=bool))):
            matrix = Z[subset]
            if filename in manifest.get("scalers", {}):
                scaler = manifest["scalers"][filename]
                names = manifest["feature_names"]
                for group in ("season", "weathersit", "weekday"):
                    idx = [i for i, col in enumerate(names) if col.startswith(group + "_")]
                    if not idx:
                        raise ValueError(f"No indicator columns for {group}")
                    # invert training scaling to check the exact one-hot identity
                    reconstructed = (matrix[:, idx] * np.asarray(scaler["scale"])[idx]
                                     + np.asarray(scaler["mean"])[idx])
                    err = np.max(np.abs(reconstructed.sum(axis=1) - 1))
                    if err > 1e-12:
                        raise ValueError(f"Failed {group} one-hot identity on {filename}: {err}")
                    relations.append(dict(seed=seed, seen_fraction=seen, matrix=name,
                                          group=group, max_abs_residual=err,
                                          columns=" ".join(names[i] for i in idx)))
            row = design_diagnostic(matrix, name)
            row.update(seen_fraction=seen, seed=seed)
            design_rows.append(row)
            beta, res, rank, s = np.linalg.lstsq(matrix, y[subset], rcond=None)
            residual = float(np.linalg.norm(matrix @ beta - y[subset]))
            ynorm = float(np.linalg.norm(y[subset]))
            ols_rows.append({"seen_fraction": seen, "seed": seed,
                             "target": "seen" if name == "X_seen_aug" else "all",
                             "finite": bool(np.isfinite(beta).all()),
                             "theta_norm": float(np.linalg.norm(beta)),
                             "residual_norm": residual, "residual_squared": residual ** 2,
                             "relative_residual": residual / ynorm if ynorm else np.nan,
                             "lstsq_rank": int(rank)})
    if relations:
        pd.DataFrame(relations).to_csv(destination / "one_hot_relations.csv", index=False)
    df_design = pd.DataFrame(design_rows)
    df_ols = pd.DataFrame(ols_rows)
    if relations:
        seen_design = df_design[df_design.matrix == "X_seen_aug"]
        ranks = ", ".join(str(int(r)) for r in sorted(seen_design["rank"].unique()))
        dims = seen_design["n_cols"].unique()
        if len(dims) != 1:
            raise ValueError("Inconsistent design dimension across runs")
        paragraphs = [f"The augmented seen-data designs had {dims[0]} columns. "
                      f"The observed numerical ranks were {ranks}."]
        for (fraction, mode), group in metrics.groupby(["seen", "mode"]):
            coverage = group["objective_usable_fraction"]
            paragraphs.append(
                f"At {fraction:.0%} seen ({mode} gossip), the final usable fraction "
                f"averaged {coverage.mean():.1%} across {len(group)} seeds "
                f"(range {coverage.min():.1%} to {coverage.max():.1%}).")
        (out / "bs_rank_and_coverage.tex").write_text(
            (" ".join(paragraphs) + "\n").replace("%", r"\%"))
    df_design.to_csv(destination / "x_aug_diagnostics.csv", index=False)
    df_ols.to_csv(destination / "centralized_ols.csv", index=False)
    metrics.rename(columns={"seen": "seen_fraction"}).to_csv(
        destination / "distributed_vs_central.csv", index=False)
    lines = [f"Bike Sharing diagnostics for {root}",
             "Numerical eligibility uses spectral condition threshold and successful solve.",
             "Objective results omit nonfinite or ineligible agent solutions, and report usable fraction.",
             "All objectives compare against centralized OLS on the SAME design matrix.", ""]
    if not df_design.empty:
        lines.append("Design ranks and condition numbers:")
        for row in df_design.itertuples():
            lines.append(f"  seen={row.seen_fraction:.2f} seed={row.seed} {row.matrix}: "
                         f"rank={row.rank}/{row.n_cols}, cond={row.cond_full:.4g}")
    if not df_ols.empty:
        lines.append("\nCentralized minima (squared residuals):")
        for row in df_ols.itertuples():
            lines.append(f"  seen={row.seen_fraction:.2f} seed={row.seed} "
                         f"Z={row.target}, min ||Z theta-y||² = {row.residual_squared:.6g}")
    lines.append("\nFinal distributed-vs-centralized (mean eligible-agent SSE):")
    for row in metrics.itertuples():
        lines.append(f"  {row.run}: stable={row.final_stable_fraction:.3f}, "
                     f"seen SSE={row.agent_sse_mean_seen:.6g}, central min={row.central_min_seen:.6g}, "
                     f"excess={row.objective_gap_mean_seen:.6g}")
    (destination / "summary.txt").write_text("\n".join(lines) + "\n")
    print(f"Saved diagnostics in {destination}")



def analyze_mass_drift_jumps(traces, metrics, config, out, radius=5):
    if not {"run", "step", "mass_drift"}.issubset(traces.columns):
        print("Skipping mass-drift diagnostics: no mass_drift checkpoint column.")
        return

    summary, windows, candidates = [], [], []
    metric_lookup = metrics.set_index("run") if "run" in metrics else pd.DataFrame()
    series_columns = ("mass_drift", "E_M", "E_B", "theta_error",
                      "stable_fraction", "objective_usable_fraction",
                      "cond_mean", "cond_max")

    for run_name, rows in traces.groupby("run", sort=True):
        rows = rows.sort_values("step").reset_index(drop=True).copy()
        steps = pd.to_numeric(rows["step"], errors="coerce").to_numpy(float)
        drift = pd.to_numeric(rows["mass_drift"], errors="coerce").to_numpy(float)
        differences = np.full(len(rows), np.nan)
        if len(rows) > 1:
            differences[1:] = np.diff(drift)
        valid = np.isfinite(differences) & np.isfinite(steps)
        record = {"run": run_name,
                  "seen": float(rows["seen"].iloc[0]) if "seen" in rows else np.nan,
                  "mode": rows["mode"].iloc[0] if "mode" in rows else "unknown",
                  "seed": rows["seed"].iloc[0] if "seed" in rows else np.nan,
                  "checkpoints": len(rows),
                  "jump_status": "insufficient_valid_checkpoints",
                  "jump_step": np.nan, "previous_step": np.nan,
                  "jump_delta": np.nan, "previous_mass_drift": np.nan,
                  "mass_drift_at_jump": np.nan, "relative_jump": np.nan,
                  "jump_at_last_checkpoint": False,
                  "active_agents_source": "not_recorded",
                  "active_agents_expected": np.nan,
                  "smallest_singular_value": np.nan}
        if run_name in metric_lookup.index:
            meta = metric_lookup.loc[run_name]
            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]
            n_agents = pd.to_numeric(meta.get("n_agents", np.nan), errors="coerce")
        else:
            n_agents = np.nan
        if config.get("stop_on_theta_change") is False and np.isfinite(n_agents):
            record["active_agents_expected"] = int(n_agents)
            record["active_agents_source"] = "inferred_all_active_from_disabled_stopping"
        if not valid.any():
            summary.append(record)
            continue

        positions = np.flatnonzero(valid)
        index = int(positions[np.argmax(differences[positions])])
        delta = float(differences[index])
        previous_drift = float(drift[index-1])
        record.update(jump_status="increase" if delta > 0 else "no_positive_increase",
                      jump_step=int(steps[index]), previous_step=int(steps[index-1]),
                      jump_delta=delta, previous_mass_drift=previous_drift,
                      mass_drift_at_jump=float(drift[index]),
                      relative_jump=delta / previous_drift if previous_drift > 0 else np.nan,
                      jump_at_last_checkpoint=bool(index == len(rows)-1))
        for name in series_columns:
            if name not in rows:
                continue
            before = pd.to_numeric(rows[name].iloc[index-1], errors="coerce")
            after = pd.to_numeric(rows[name].iloc[index], errors="coerce")
            record[f"before_{name}"] = before
            record[f"after_{name}"] = after
            record[f"change_{name}"] = after - before
        summary.append(record)

        window_start = max(0, index-radius)
        window_end = min(len(rows), index+radius+1)
        nearby = rows.iloc[window_start:window_end].copy()
        nearby["drift_change_since_previous_checkpoint"] = differences[window_start:window_end]
        nearby["is_largest_drift_increase_checkpoint"] = (nearby.index == index)
        nearby["selected_jump_step"] = int(steps[index])
        windows.append(nearby)
        candidates.append((delta, run_name, rows, index))

    summary_path = out / "mass_drift_jump_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    if windows:
        pd.concat(windows, ignore_index=True).to_csv(out / "mass_drift_diagnostics.csv", index=False)
    else:
        pd.DataFrame(columns=["run", "step", "mass_drift",
                              "drift_change_since_previous_checkpoint",
                              "is_largest_drift_increase_checkpoint",
                              "selected_jump_step"]).to_csv(out / "mass_drift_diagnostics.csv", index=False)

    if not candidates:
        print(f"Mass-drift diagnostics saved: {summary_path} (no comparable checkpoints)")
        return
    individual = out / "mass_drift_run_plots"
    individual.mkdir(exist_ok=True)
    for delta, run_name, rows, index in candidates:
        start, end = max(0, index-radius), min(len(rows), index+radius+1)
        nearby = rows.iloc[start:end]
        fig, axs = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
        panels = (
            (axs[0, 0], ("mass_drift",), "Mass drift"),
            (axs[0, 1], ("E_M", "E_B"), "Statistic disagreement"),
            (axs[1, 0], ("stable_fraction", "objective_usable_fraction"), "Numerical usability"),
            (axs[1, 1], ("cond_mean", "cond_max"), "Local condition numbers"),
            (axs[2, 0], ("theta_error",), "Parameter error"),
        )
        for ax, fields, heading in panels:
            any_plotted = False
            for field in fields:
                if field not in nearby:
                    continue
                x = pd.to_numeric(nearby["step"], errors="coerce").to_numpy(float)
                y = pd.to_numeric(nearby[field], errors="coerce").to_numpy(float)
                good = np.isfinite(x) & np.isfinite(y)
                if not good.any():
                    continue
                ax.plot(x[good], y[good], "o-", label=field, markersize=4)
                any_plotted = True
            ax.axvline(rows.step.iloc[index], color="tab:red", linestyle="--", linewidth=1,
                       label="selected jump")
            ax.set_title(heading)
            ax.grid(alpha=.25)
            if any_plotted:
                if heading == "Numerical usability":
                    ax.set_ylim(-.05, 1.05)
                elif heading in ("Statistic disagreement", "Local condition numbers", "Parameter error"):
                    values = []
                    for line in ax.lines[:-1]:
                        values.extend(np.asarray(line.get_ydata(), dtype=float).tolist())
                    finite = np.asarray(values, dtype=float)
                    finite = finite[np.isfinite(finite)]
                    if len(finite) and np.all(finite > 0):
                        ax.set_yscale("log")
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "Not recorded / no finite values", ha="center", va="center",
                        transform=ax.transAxes)
        differences = pd.to_numeric(rows["mass_drift"], errors="coerce").diff()
        local_delta = differences.iloc[start:end]
        axs[2, 1].plot(nearby["step"], local_delta, "o-", markersize=4)
        axs[2, 1].axhline(0, color="black", linewidth=.7)
        axs[2, 1].axvline(rows.step.iloc[index], color="tab:red", linestyle="--", linewidth=1)
        axs[2, 1].set_title("Change in drift since prior checkpoint")
        axs[2, 1].grid(alpha=.25)
        for ax in axs[2, :]:
            ax.set_xlabel("Gossip interactions")
        fig.suptitle(f"{run_name} | largest logged drift increase={delta:.3g}\n"
                     "All panels use saved checkpoints; singular values not available.")
        fig.tight_layout()
        image_path = individual / f"{run_name}.png"
        fig.savefig(image_path, dpi=170)
        if (delta, run_name) == max((v, n) for v, n, _, _ in candidates):
            fig.savefig(out / "mass_drift_diagnostics.png", dpi=200)
        plt.close(fig)
    best = max(candidates, key=lambda r: r[0])
    print(f"Mass-drift diagnostics: {len(candidates)} runs, largest logged jump "
          f"{best[0]:.4g} in {best[1]}; CSV and plots in {out}")

def verify_checkpoint_exports(root, traces, out):
    for point in traces.itertuples():
        path = root / point.run / "agent_objectives" / f"step_{point.step}.npz"
        with np.load(path) as raw:
            count = len(raw["agent"])
            total = int(raw["n_agents"])
            if count != point.objective_usable_count or total != point.n_agents:
                raise ValueError(f"{path}: inconsistent agent count")
            np.testing.assert_allclose(count / total, point.objective_usable_fraction)
            for tag in ("seen", "all"):
                sse = raw[f"sse_{tag}"]
                gap = raw[f"signed_gap_{tag}"]
                minimum = raw[f"central_min_{tag}"]
                np.testing.assert_array_equal(gap, sse - minimum)
                np.testing.assert_allclose(minimum, getattr(point, f"central_min_{tag}"))
                for field, values in (("agent_sse_mean", sse), ("objective_gap_mean", gap)):
                    np.testing.assert_allclose(values.mean() if count else np.nan,
                                               getattr(point, f"{field}_{tag}"), equal_nan=True)
    traces.to_csv(out / "bs_checkpoint_audit.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_dir", nargs="?", type=Path)
    args = parser.parse_args()
    if args.results_dir is None:
        candidates = sorted(p for p in (HERE / "results").glob("exp_BS_*") if p.is_dir())
        if not candidates:
            parser.error(f"No experiment folders under {HERE / 'results'}")
        root = candidates[-1]
    else:
        root = args.results_dir.expanduser().resolve()
    print(f"Analyzing: {root}")
    cfg_path = root / "_config.json"
    cfg = json.loads(cfg_path.read_text()) if cfg_path.is_file() else {}
    traces, metrics, _agents = load_runs(root)
    out = root / "analysis_figures"
    out.mkdir(exist_ok=True)
    p = cfg.get("p_aug") or metrics.get("p_aug", pd.Series([np.nan])).dropna().iloc[0]
    if not np.isfinite(float(p)):
        raise ValueError("Unknown augmented dimension; _config.json is required")
    p = int(p)
    print(f"Loaded {len(metrics)} runs; p_aug={p}; full-state cost={communication_cost('full', p)}, "
          f"coordinate cost=2 scalars per interaction")
    if (root / "designs").is_dir():
        verify_checkpoint_exports(root, traces, out)
    make_panels(traces, out)
    make_mode_seen_comparisons(traces, out)
    make_communication_plot(traces, p, out)
    make_prediction_plot(metrics, out)
    make_objective_plots(traces, metrics, out)
    make_stability_tables(metrics, out)
    analyze_mass_drift_jumps(traces, metrics, cfg, out)
    tolerance_crossing_table(traces, p, out)
    slope_fits(traces, out)
    if "final_stable_fraction" in metrics:
        columns = [c for c in ("seen_RMSE", "unseen_RMSE", "seen_MAE", "unseen_MAE",
                               "seen_R2", "unseen_R2", "rel_error_seen", "rel_error_global",
                               "final_stable_fraction") if c in metrics]
        if columns:
            metrics.groupby(["seen", "mode"])[columns].agg(["mean", "std"]).to_csv(
                out / "bs_metrics_summary.csv")
    analyze_diagnostics(root, cfg, metrics, out)
    print(f"Analysis complete: {out}")


if __name__ == "__main__":
    main()
