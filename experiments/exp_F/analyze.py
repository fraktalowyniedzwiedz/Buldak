import os, sys, glob, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def write_final_stability_report(records, output_dir, group_columns):
    from pathlib import Path

    if not records:
        print("No final-stability records available")
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

def get_metric_value(met, key, default=None):
    parts = key.split('.')
    val = met
    for p in parts:
        if isinstance(val, dict) and p in val:
            val = val[p]
        else:
            return default
    return val

def consolidate_F(base_dir):
    folders = sorted(glob.glob(os.path.join(base_dir, "NoiseSep_*")))
    if not folders:
        print("No NoiseSep_* folders found; nothing to consolidate.")
        return

    conv_frames = []
    per_run = []

    for folder in folders:
        name = os.path.basename(folder)
        parts = name.split('_')
        try:
            noise = float(parts[1])
            A = int(parts[2][1:])
            D = int(parts[3][1:])
            seed = int(parts[4][4:])
        except (IndexError, ValueError):
            print(f"Skipping {folder}: unrecognised folder name.")
            continue

        conv_path = os.path.join(folder, "convergence.csv")
        met_path = os.path.join(folder, "metrics.json")

        if os.path.exists(conv_path):
            conv = pd.read_csv(conv_path)
            conv.insert(0, 'noise', noise)
            conv.insert(1, 'seed', seed)
            conv.insert(2, 'A', A)
            conv.insert(3, 'D', D)
            conv_frames.append(conv)

        if os.path.exists(met_path):
            with open(met_path, 'r') as f:
                met = json.load(f)
            met['noise'] = noise
            met['seed'] = seed
            met['A'] = A
            met['D'] = D
            per_run.append(met)

    if conv_frames:
        conv_all = pd.concat(conv_frames, ignore_index=True)
        conv_all = conv_all.sort_values(['noise', 'seed', 'step']).reset_index(drop=True)
        conv_out = os.path.join(base_dir, 'convergence.csv')
        conv_all.to_csv(conv_out, index=False)
        print(f"Saved {conv_out}  ({len(conv_all)} rows, "
              f"{conv_all['noise'].nunique()} noise levels, "
              f"{conv_all['seed'].nunique()} seeds)")
    else:
        print("No convergence.csv files found; skipping consolidation.")

    if per_run:
        runs_df = pd.DataFrame(per_run)

        flat_rows = []
        for rec in per_run:
            row = {
                'noise': rec['noise'],
                'seed': rec['seed'],
                'A': rec['A'],
                'D': rec['D'],
                'rel_error_seen': rec.get('rel_error_seen'),
                'rel_error_global': rec.get('rel_error_global'),
                'final_stable_fraction': rec.get('final_stable_fraction'),
                'steps': rec.get('steps'),
            }
            for split in ('Seen', 'Unseen'):
                if split in rec and isinstance(rec[split], dict):
                    for k, v in rec[split].items():
                        row[f'{split}_{k}'] = v
            flat_rows.append(row)

        flat = pd.DataFrame(flat_rows)

        agg_cols = [c for c in flat.columns
                    if c not in ('noise', 'seed', 'A', 'D')]

        by_noise = {}
        for noise, grp in flat.groupby('noise'):
            entry = {'n_seeds': int(len(grp))}
            for c in agg_cols:
                vals = grp[c].dropna()
                if len(vals) == 0:
                    continue
                entry[c] = {
                    'mean': float(vals.mean()),
                    'std':  float(vals.std(ddof=1)) if len(vals) > 1 else 0.0,
                }
            by_noise[f"{noise:g}"] = entry

        consolidated = {
            'meta': {
                'experiment': 'F',
                'description': 'Noise separation with noise-free control',
                'A': int(flat['A'].iloc[0]),
                'D': int(flat['D'].iloc[0]),
                'noise_levels': sorted(flat['noise'].unique().tolist()),
                'seeds': sorted(flat['seed'].unique().tolist()),
                'n_runs': int(len(flat)),
                'source': os.path.basename(base_dir),
            },
            'runs': sorted(
                [
                    {
                        'noise': float(r['noise']),
                        'seed': int(r['seed']),
                        'rel_error_seen': r.get('rel_error_seen'),
                        'rel_error_global': r.get('rel_error_global'),
                        'final_stable_fraction': r.get('final_stable_fraction'),
                        'steps': r.get('steps'),
                        'Seen':   {k: v for k, v in
                                   (per_run[i].get('Seen') or {}).items()},
                        'Unseen': {k: v for k, v in
                                   (per_run[i].get('Unseen') or {}).items()},
                    }
                    for i, r in flat.iterrows()
                ],
                key=lambda d: (d['noise'], d['seed']),
            ),
            'by_noise': by_noise,
        }

        met_out = os.path.join(base_dir, 'metrics.json')
        with open(met_out, 'w') as f:
            json.dump(consolidated, f, indent=2)
        print(f"Saved {met_out}  ({len(per_run)} runs aggregated "
              f"over {len(by_noise)} noise levels)")
    else:
        print("No metrics.json files found; skipping consolidation.")


def main():
    if len(sys.argv) > 1:
        base_dir = sys.argv[1]
    else:
        exp_dir = os.path.dirname(os.path.abspath(__file__))
        all_dirs = sorted(glob.glob(os.path.join(exp_dir, "results", "exp_F_*")))
        if not all_dirs:
            print("No exp_F_* directories found.")
            return
        base_dir = all_dirs[-1]
    print(f"Analysing results in: {base_dir}")

    folders = glob.glob(os.path.join(base_dir, "NoiseSep_*"))
    records = []
    stability_records = []
    conv_dfs = {}

    for folder in folders:
        name = os.path.basename(folder)
        parts = name.split('_')
        noise = float(parts[1])
        seed = int(parts[4][4:])
        met_path = os.path.join(folder, "metrics.json")
        conv_path = os.path.join(folder, "convergence.csv")
        if not os.path.isfile(conv_path):
            stability_records.append({
                "run": name, "noise": noise, "seed": seed,
                "last_logged_step": np.nan, "final_Etheta": np.nan,
                "final_stable_fraction": np.nan, "stability_source": "missing",
            })
            continue
        conv = pd.read_csv(conv_path)
        met = {}
        if os.path.isfile(met_path):
            with open(met_path, 'r') as f:
                met = json.load(f)
        if conv.empty:
            last_step, last_theta, fraction = np.nan, np.nan, np.nan
        else:
            last = conv.sort_values('step').iloc[-1]
            last_step, last_theta = last['step'], last.get('theta_error', np.nan)
            fraction = last.get('stable_fraction', np.nan)
        source = 'convergence.csv' if 'stable_fraction' in conv.columns else 'missing'
        if 'final_stable_fraction' in met:
            fraction = met['final_stable_fraction']
            source = 'metrics.json'
        stability_records.append({
            "run": name, "noise": noise, "seed": seed,
            "last_logged_step": last_step, "final_Etheta": last_theta,
            "final_stable_fraction": fraction, "stability_source": source,
        })
        if conv.empty:
            continue
        if not os.path.isfile(met_path):
            continue
        key = noise
        if key not in conv_dfs:
            conv_dfs[key] = []
        conv_dfs[key].append(conv)

        final_E_M = conv['E_M'].iloc[-1]
        final_E_B = conv['E_B'].iloc[-1]
        final_theta_err = conv['theta_error'].iloc[-1]
        rel_err_global = met.get('rel_error_global', np.nan)
        test_rmse = get_metric_value(met, 'Unseen.RMSE')
        records.append({
            'noise': noise,
            'seed': seed,
            'final_E_M': final_E_M,
            'final_E_B': final_E_B,
            'final_theta_err': final_theta_err,
            'rel_err_global': rel_err_global,
            'test_RMSE': test_rmse
        })

    write_final_stability_report(stability_records, base_dir, ["noise"])
    if not records:
        print("No data found.")
        return

    df = pd.DataFrame(records)
    summary = df.groupby('noise').agg({
        'final_E_M': ['mean', 'std'],
        'final_E_B': ['mean', 'std'],
        'final_theta_err': ['mean', 'std'],
        'rel_err_global': ['mean', 'std'],
        'test_RMSE': ['mean', 'std']
    }).reset_index()
    summary.columns = ['noise', 'E_M_mean', 'E_M_std', 'E_B_mean', 'E_B_std',
                       'theta_mean', 'theta_std', 'rel_global_mean', 'rel_global_std',
                       'test_RMSE_mean', 'test_RMSE_std']
    print("\nNoise separation summary:")
    print(summary.to_string(index=False))

    plt.figure(figsize=(10,7))
    cmap = plt.cm.viridis
    noise_values = sorted(conv_dfs.keys())
    norm = plt.Normalize(vmin=min(noise_values), vmax=max(noise_values))
    for noise in noise_values:
        dfs = conv_dfs[noise]
        all_conv = pd.concat(dfs, ignore_index=False)
        grouped = all_conv.groupby('step').agg({'theta_error': ['mean', 'std']}).reset_index()
        grouped.columns = ['step', 'theta_mean', 'theta_std']
        color = cmap(norm(noise))
        plt.plot(grouped['step'], grouped['theta_mean'], color=color, label=f'σ={noise}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Gossip step')
    plt.ylabel('Theta error (Eθ)')
    plt.title('Convergence of parameter error for different noise levels')
    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
    plt.tight_layout()
    outfig1 = os.path.join(base_dir, 'convergence_trajectories.png')
    plt.savefig(outfig1, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved {outfig1}")

    fig, ax1 = plt.subplots(figsize=(10,6))
    ax1.set_xlabel('Noise level (σ)')
    ax1.set_ylabel('Final theta error (consensus)', color='tab:blue')
    ax1.errorbar(summary['noise'], summary['theta_mean'], yerr=summary['theta_std'],
                 fmt='o-', color='tab:blue', capsize=5, label='θ error')
    ax1.set_yscale('log')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, linestyle='--', alpha=0.3)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Test RMSE (prediction)', color='tab:red')
    ax2.errorbar(summary['noise'], summary['test_RMSE_mean'], yerr=summary['test_RMSE_std'],
                 fmt='s-', color='tab:red', capsize=5, label='Test RMSE')
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    plt.title('Consensus error vs prediction error as noise varies')
    plt.xscale('log')
    fig.tight_layout()
    outfig2 = os.path.join(base_dir, 'final_theta_vs_test_rmse.png')
    plt.savefig(outfig2, dpi=300)
    plt.close()
    print(f"Saved {outfig2}")

    plt.figure(figsize=(8,5))
    plt.errorbar(summary['noise'], summary['E_M_mean'], yerr=summary['E_M_std'],
                 fmt='o-', label='E_M', capsize=5)
    plt.errorbar(summary['noise'], summary['E_B_mean'], yerr=summary['E_B_std'],
                 fmt='s-', label='E_B', capsize=5)
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Noise level (σ)')
    plt.ylabel('Statistic error (final)')
    plt.title('Statistic consensus errors are independent of noise')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    outfig3 = os.path.join(base_dir, 'statistic_errors_vs_noise.png')
    plt.savefig(outfig3, dpi=300)
    plt.close()
    print(f"Saved {outfig3}")

    summary.to_csv(os.path.join(base_dir, 'noise_summary.csv'), index=False)
    consolidate_F(base_dir)
    print("Analysis complete.")

if __name__ == "__main__":
    main()



