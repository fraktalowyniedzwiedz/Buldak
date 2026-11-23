import os
import argparse
import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)


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
    parser = argparse.ArgumentParser(description='Analyze exp_G results')
    parser.add_argument('results_dir', nargs='?', help='Run directory (default: latest)')
    args = parser.parse_args()
    results_base = os.path.join(os.path.dirname(__file__), 'results')
    if args.results_dir:
        outdir = os.path.abspath(args.results_dir)
    else:
        if not os.path.isdir(results_base):
            print('No exp_G results found.')
            return
        exp_dirs = sorted(d for d in os.listdir(results_base)
                          if d.startswith('exp_G_') and os.path.isfile(os.path.join(results_base, d, 'results.json')))
        if not exp_dirs:
            print('No exp_G_* directories with results.json found.')
            return
        outdir = os.path.join(results_base, exp_dirs[-1])
    print(f"Analyzing results from {outdir}")

    with open(os.path.join(outdir, 'results.json'), 'r') as f:
        results = json.load(f)

    profiles = {r.get('profile', 'archive') for r in results}
    if len(profiles) != 1 or not profiles <= {'thesis', 'archive'}:
        raise ValueError('Expected results from one experiment G profile')
    thesis = profiles == {'thesis'}
    metric = 'gossip_final_mean_agent_error' if thesis else 'gossip_final_err'
    print(f"Gossip bar uses {metric}")

    stability_records = []
    for result in results:
        seed = result["seed"]
        conv_file = os.path.join(outdir, f"convergence_seed{seed}.csv")
        step, last_theta, theta_source = np.nan, np.nan, "missing"
        if os.path.isfile(conv_file):
            conv = pd.read_csv(conv_file)
            if not conv.empty:
                last = conv.sort_values("step").iloc[-1]
                step = last.get("step", np.nan)
                last_theta = last.get("theta_error", np.nan)
                theta_source = "convergence.csv"
        stability_records.append({
            "run": f"seed{seed}", "method": "gossip", "seed": seed,
            "last_logged_step": step,
            "final_stable_fraction": result.get("stable_fraction", np.nan),
            "stability_source": "results.json",
            "final_Etheta": last_theta,
            "Etheta_source": theta_source,
            "gossip_final_err_of_mean": result.get("gossip_final_err", np.nan),
        })
    write_final_stability_report(stability_records, outdir, ["method"])

    print("\nRaw results:")
    for r in results:
        print(f"Seed {r['seed']}: err_unweighted={r['err_unweighted']:.4e}, "
              f"err_weighted={r['err_weighted']:.4e}, "
              f"{metric}={r[metric]:.4e}, "
              f"stable_fraction={r['stable_fraction']:.3f}")

    seeds = [r['seed'] for r in results]
    err_unweighted = [r['err_unweighted'] for r in results]
    err_weighted = [r['err_weighted'] for r in results]
    err_gossip = [r[metric] for r in results]

    for name, values in [('unweighted', err_unweighted), ('weighted', err_weighted), ('gossip', err_gossip)]:
        if not np.isfinite(values).all():
            print(f'Warning: non-finite {name} errors; omitting them from aggregates')

    data = {
        'Method': ['Unweighted local avg', 'Weighted local avg', 'Gossip (sufficient stats)'],
        'Mean error': [np.nanmean(err_unweighted), np.nanmean(err_weighted), np.nanmean(err_gossip)],
        'Std error': [np.nanstd(err_unweighted), np.nanstd(err_weighted), np.nanstd(err_gossip)]
    }
    df = pd.DataFrame(data)
    print("\nComputed DataFrame:")
    print(df)

    df.to_csv(os.path.join(outdir, 'error_comparison.csv'), index=False)

    plt.figure(figsize=(8, 5))
    methods = df['Method']
    means = df['Mean error']
    stds = df['Std error']
    x = np.arange(len(methods))
    bars = plt.bar(x, means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    plt.xticks(x, methods, rotation=15)
    plt.ylabel('Relative parameter error')
    plt.title('Heterogeneity / Simpson: comparison of aggregation methods')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.ylim(bottom=0)
    for bar, val in zip(bars, means):
        if not np.isnan(val):
            plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                     f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'heterogeneity_comparison.png'), dpi=300)
    plt.close()

    conv_dfs = []
    for seed in seeds:
        fname = os.path.join(outdir, f'convergence_seed{seed}.csv')
        if os.path.exists(fname):
            conv = pd.read_csv(fname)
            conv_dfs.append(conv)
    if conv_dfs:
        checkpoints = pd.concat(conv_dfs, ignore_index=True).groupby('step')
        steps = np.asarray(sorted(checkpoints.groups))
        mean_curve = checkpoints['theta_error'].mean().to_numpy()
        std_curve = checkpoints['theta_error'].std(ddof=0).to_numpy()
        mean_stable = checkpoints['stable_fraction'].mean().to_numpy()

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
        ax1.plot(steps, mean_curve, label='Mean theta error')
        ax1.fill_between(steps, mean_curve-std_curve, mean_curve+std_curve, alpha=0.2)
        ax1.set_yscale('log')
        ax1.set_ylabel('Relative theta error')
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.legend()

        ax2.plot(steps, mean_stable, label='Stable fraction')
        ax2.set_xlabel('Gossip steps')
        ax2.set_ylabel('Stable fraction')
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(outdir, 'convergence_avg.png'), dpi=300)
        plt.close()
        print(f"Saved convergence plot to {outdir}/convergence_avg.png")

    print("Analysis complete.")

if __name__ == "__main__":
    main()