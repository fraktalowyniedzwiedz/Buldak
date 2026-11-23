import os, sys, glob, re, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
import matplotlib

DELTAS = [1e-2, 1e-4, 1e-6, 1e-8, 1e-10]

SEEDS_PER_KAPPA = 10
MIN_SEED_FRACTION = 0.8

SLOPE_FIT_MAX_E_THETA = 0.1


def parse_kappa(folder_name):
    m = re.search(r'kappa([0-9.eE+-]+)_', folder_name)
    return float(m.group(1)) if m else None


def parse_seed(folder_name):
    m = re.search(r'seed(\d+)', folder_name)
    return int(m.group(1)) if m else -1


def find_crossing(conv, delta):
    """first checkpoint where E_M <= delta, log-interpolated"""
    E_M = conv['E_M'].to_numpy(dtype=float)
    valid = np.isfinite(E_M) & (E_M > 0)
    if not valid.any():
        return None
    idx = np.where(valid)[0]
    below = np.where(E_M[idx] <= delta)[0]
    if len(below) == 0:
        return None
    
    available_cols = ['E_M', 'E_B', 'theta_error', 'cond_mean']
    if 'E_M2' in conv.columns:
        available_cols.append('E_M2')

    i_first = below[0]
    if i_first == 0:
        j = idx[0]
        out = {c: float(conv[c].iloc[j]) for c in available_cols}
        out['step'] = int(conv['step'].iloc[j])
        return out

    i0, i1 = idx[i_first - 1], idx[i_first]
    EM0, EM1 = E_M[i0], E_M[i1]
    t = (np.log(delta) - np.log(EM0)) / (np.log(EM1) - np.log(EM0))

    def interp(col):
        v0, v1 = float(conv[col].iloc[i0]), float(conv[col].iloc[i1])
        if v0 > 0 and v1 > 0:
            return float(np.exp(np.log(v0) + t * (np.log(v1) - np.log(v0))))
        return float(v0 + t * (v1 - v0))

    out = {c: interp(c) for c in available_cols}
    out['step'] = int(conv['step'].iloc[i0]
                      + t * (conv['step'].iloc[i1] - conv['step'].iloc[i0]))
    return out



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

def analyze_gossip(base_dir):
    print(f"Analysing results in: {base_dir}")

    folders = sorted(glob.glob(os.path.join(base_dir, "Amp_*")))
    if not folders:
        print("No Amp_* run folders found.")
        return

    records = []
    stability_records = []
    for folder in folders:
        name = os.path.basename(folder)
        kappa = parse_kappa(name)
        if kappa is None:
            continue
        conv_path = os.path.join(folder, "convergence.csv")
        if not os.path.exists(conv_path):
            continue
        conv = pd.read_csv(conv_path)
        seed = parse_seed(name)
        if conv.empty:
            print(f"  WARNING: empty convergence log in {name}; final stability unknown.")
            stability_records.append({
                "run": name, "kappa_target": kappa, "seed": seed,
                "last_logged_step": np.nan, "final_stable_fraction": np.nan,
                "stability_source": "missing", "final_Etheta": np.nan,
            })
            continue
        last = conv.sort_values("step").iloc[-1]
        fraction = last.get("stable_fraction", np.nan)
        fraction_source = "convergence.csv" if "stable_fraction" in conv else "missing"
        metrics_file = os.path.join(folder, "metrics.json")
        if os.path.isfile(metrics_file):
            with open(metrics_file) as file:
                metrics = json.load(file)
            if "final_stable_fraction" in metrics:
                fraction = metrics["final_stable_fraction"]
                fraction_source = "metrics.json"
        stability_records.append({
            "run": name, "kappa_target": kappa, "seed": seed,
            "last_logged_step": int(last["step"]),
            "final_stable_fraction": fraction,
            "stability_source": fraction_source,
            "final_Etheta": last.get("theta_error", np.nan),
        })

        reached = {}
        for delta in DELTAS:
            cross = find_crossing(conv, delta)
            if cross is None:
                reached[delta] = False
                continue
            reached[delta] = True
            
            rec = dict(
                kappa_target=kappa, seed=seed, delta=delta,
                step_at_crossing=cross['step'],
                E_M_at_crossing=cross['E_M'],
                E_B_at_crossing=cross['E_B'],
                E_theta_at_crossing=cross['theta_error'],
                kappa_empirical=cross['cond_mean'],
            )
            if 'E_M2' in cross:
                rec['E_M2_at_crossing'] = cross['E_M2']
            
            records.append(rec)
            
        if not reached[min(DELTAS)]:
            print(f"  WARNING: {name} did not reach E_M = {min(DELTAS):g}.")

    write_final_stability_report(
        stability_records, os.path.join(os.path.dirname(base_dir), "analysis", "gossip"),
        ["kappa_target"])
    if not records:
        print("No usable records.")
        return

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(base_dir, 'amplification_runs.csv'), index=False)

    kappa_check = (df.groupby('kappa_target')
                     .agg(kappa_emp_mean=('kappa_empirical', 'mean'),
                          kappa_emp_std=('kappa_empirical', 'std'),
                          n=('kappa_empirical', 'count'))
                     .reset_index())
    print("\nPrescribed vs empirical kappa:")
    print(kappa_check.to_string(index=False))
    kappa_check.to_csv(os.path.join(base_dir, 'kappa_check.csv'), index=False)

    agg_dict = {
        'E_theta_mean': ('E_theta_at_crossing', 'mean'),
        'E_theta_std': ('E_theta_at_crossing', 'std'),
        'E_M_mean': ('E_M_at_crossing', 'mean'),
        'E_B_mean': ('E_B_at_crossing', 'mean'),
        'step_mean': ('step_at_crossing', 'mean'),
        'n': ('E_theta_at_crossing', 'count'),
    }
    if 'E_M2_at_crossing' in df.columns:
        agg_dict['E_M2_mean'] = ('E_M2_at_crossing', 'mean')

    agg = df.groupby(['kappa_target', 'delta']).agg(**agg_dict).reset_index()

    before = len(agg)
    agg = agg[agg['n'] >= MIN_SEED_FRACTION * SEEDS_PER_KAPPA].reset_index(drop=True)
    print(f"\nFiltered {(before - len(agg))} low-coverage (kappa, delta) points "
          f"(kept n >= {MIN_SEED_FRACTION * SEEDS_PER_KAPPA:g}).")

    agg['sqrt_EM'] = np.sqrt(agg['E_M_mean'])
    if 'E_M2_mean' in agg.columns:
        agg['sqrt_EM2'] = np.sqrt(agg['E_M2_mean'])

    agg.to_csv(os.path.join(base_dir, 'amplification_aggregate.csv'), index=False)
    print("\nAggregate over seeds:")
    print(agg.to_string(index=False))

    # Fig 1: E_theta vs prescribed kappa, one curve per delta
    fig, ax = plt.subplots(figsize=(9, 6))
    for delta in DELTAS:
        sub = agg[agg['delta'] == delta].sort_values('kappa_target')
        if sub.empty:
            continue
        ax.errorbar(sub['kappa_target'], sub['E_theta_mean'],
                    yerr=sub['E_theta_std'].fillna(0.0),
                    fmt='o-', capsize=4,
                    label=fr'$E_\theta$ at $E_M = {delta:g}$')

    sub_ref = agg[agg['delta'] == min(DELTAS)].sort_values('kappa_target')
    if len(sub_ref) >= 2:
        x0, y0 = sub_ref['kappa_target'].iloc[0], sub_ref['E_theta_mean'].iloc[0]
        xr = np.array([sub_ref['kappa_target'].iloc[0],
                       sub_ref['kappa_target'].iloc[-1]])
        ax.plot(xr, y0 * xr / x0, 'k--', alpha=0.7,
                label=r'reference slope $1$')

    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'prescribed $\kappa_2(\overline{{M}})$')
    ax.set_ylabel(r'$E_\theta$')
    ax.set_title('Conditioning amplification at fixed statistic error')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend()
    plt.tight_layout()
    out = os.path.join(base_dir, 'Etheta_vs_kappa.png')
    plt.savefig(out, dpi=300); plt.close()
    print(f"Saved {out}")

    # Fig 2: E_theta vs E_M, one curve per kappa
    fig, ax = plt.subplots(figsize=(9, 6))
    for kappa, sub in agg.groupby('kappa_target'):
        sub = sub.sort_values('E_M_mean')
        ax.errorbar(sub['E_M_mean'], sub['E_theta_mean'],
                    yerr=sub['E_theta_std'].fillna(0.0),
                    fmt='o-', capsize=4,
                    label=fr'$\kappa_2(\overline{{M}})={kappa:g}$')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_xlabel(r'$E_M$ at crossing')
    ax.set_ylabel(r'$E_\theta$')
    ax.set_title('Parameter error vs statistic error (Frobenius)')
    ax.grid(True, which='both', ls='--', alpha=0.5)
    ax.legend(fontsize=8)
    plt.tight_layout()
    out = os.path.join(base_dir, 'Etheta_vs_EM.png')
    plt.savefig(out, dpi=300); plt.close()
    print(f"Saved {out}")

    # Spectral Error (E_M2)
    if 'sqrt_EM2' in agg.columns:
        # Fig 3: E_theta vs sqrt(E_M2)
        fig, ax = plt.subplots(figsize=(9, 6))
        for kappa, sub in agg.groupby('kappa_target'):
            sub = sub.sort_values('sqrt_EM2')
            ax.errorbar(sub['sqrt_EM2'], sub['E_theta_mean'],
                        yerr=sub['E_theta_std'].fillna(0.0),
                        fmt='o-', capsize=4,
                        label=fr'$\kappa_2(\overline{{M}})={kappa:g}$')
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\sqrt{E_{M,2}}$ (RMS relative spectral error)')
        ax.set_ylabel(r'$E_\theta$')
        ax.set_title('Parameter error vs spectral statistic error')
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(base_dir, 'Etheta_vs_sqrt_EM2.png')
        plt.savefig(out, dpi=300); plt.close()
        print(f"Saved {out}")

        # Fig 4: Proportionality test (sqrt_EM2 vs sqrt_EM)
        fig, ax = plt.subplots(figsize=(9, 6))
        for kappa, sub in agg.groupby('kappa_target'):
            sub = sub.sort_values('sqrt_EM')
            ax.plot(sub['sqrt_EM'], sub['sqrt_EM2'], 'o-',
                    label=fr'$\kappa_2(\overline{{M}})={kappa:g}$')
        
        # ref line y = x
        if not agg.empty:
            x_min, x_max = agg['sqrt_EM'].min(), agg['sqrt_EM'].max()
            ax.plot([x_min, x_max], [x_min, x_max], 'k--', alpha=0.5, label='y = x')

        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel(r'$\sqrt{E_M}$ (Frobenius)')
        ax.set_ylabel(r'$\sqrt{E_{M,2}}$ (Spectral)')
        ax.set_title('Proportionality: Spectral vs Frobenius statistic error')
        ax.grid(True, which='both', ls='--', alpha=0.5)
        ax.legend(fontsize=8)
        plt.tight_layout()
        out = os.path.join(base_dir, 'sqrt_EM2_vs_sqrt_EM.png')
        plt.savefig(out, dpi=300); plt.close()
        print(f"Saved {out}")
    else:
        print("\nSkipping spectral figures: E_M2 not found in convergence.csv.")
        print("Did you run the experiment using run_with_spectral_metrics.py?")

    print("\nSlope of log E_theta vs log kappa "
          f"(restricted to E_theta < {SLOPE_FIT_MAX_E_THETA:g}; "
          "reference slope 1.0):")
    for delta, sub in agg.groupby('delta'):
        sub = sub[sub['E_theta_mean'] < SLOPE_FIT_MAX_E_THETA] \
                .sort_values('kappa_target')
        if len(sub) >= 2:
            s, _ = np.polyfit(np.log(sub['kappa_target']),
                              np.log(sub['E_theta_mean']), 1)
            print(f"  delta={delta:g}: slope={s:.3f}  (n={len(sub)})")
        else:
            print(f"  delta={delta:g}: not enough unsaturated points "
                  f"(n={len(sub)})")

    print("\nSlope of log E_theta vs log E_M "
          f"(restricted to E_theta < {SLOPE_FIT_MAX_E_THETA:g}; "
          "reference slope 0.5):")
    for kappa, sub in agg.groupby('kappa_target'):
        sub = sub[sub['E_theta_mean'] < SLOPE_FIT_MAX_E_THETA] \
                .sort_values('E_M_mean')
        if len(sub) >= 2:
            s, _ = np.polyfit(np.log(sub['E_M_mean']),
                              np.log(sub['E_theta_mean']), 1)
            print(f"  kappa={kappa:g}: slope={s:.3f}  (n={len(sub)})")
        else:
            print(f"  kappa={kappa:g}: not enough unsaturated points "
                  f"(n={len(sub)})")

    if 'sqrt_EM2' in agg.columns:
        print("\nSlope of log sqrt(E_M2) vs log sqrt(E_M) "
              "(Proportionality test; prediction: 1.0):")
        for kappa, sub in agg.groupby('kappa_target'):
            sub = sub.sort_values('sqrt_EM')
            if len(sub) >= 2:
                s, _ = np.polyfit(np.log(sub['sqrt_EM']),
                                  np.log(sub['sqrt_EM2']), 1)
                print(f"  kappa={kappa:g}: slope={s:.3f}")
            else:
                print(f"  kappa={kappa:g}: not enough points (n={len(sub)})")

matplotlib.use("Agg")
HERE = Path(__file__).resolve().parent

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('results_dir', type=Path, nargs='?', default=None)
    args = p.parse_args()
    if args.results_dir is None:
        dirs = sorted(x for x in (HERE / 'results').glob('exp_E_*') if x.is_dir() and (x / 'gossip').is_dir())
        if not dirs:
            p.error('No unified exp_E results found. Run run.py first or pass a results directory.')
        root = dirs[-1]
    else:
        root = args.results_dir.resolve()
    out = root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    gossip = root / 'gossip'
    if any(gossip.glob('Amp_*/convergence.csv')):
        analyze_gossip(str(gossip))
        gout = out / 'gossip'
        gout.mkdir(exist_ok=True)
        for file in gossip.iterdir():
            if file.is_file() and file.suffix.lower() in ('.png', '.csv'):
                file.replace(gout / file.name)
        print('Gossip analysis:', gout)
    else:
        print('No gossip convergence files found; skipping gossip analysis.')
    perturb = root / 'perturbation' / 'perturbation_aggregate.csv'
    if perturb.exists():
        df = pd.read_csv(perturb)
        df.to_csv(out / 'perturbation_aggregate.csv', index=False)
        fig, ax = plt.subplots(figsize=(8,5.5))
        for eps, sub in df.groupby('eps_M'):
            sub = sub.sort_values('kappa_M')
            valid = (sub['E_theta_mean'] > 0) & np.isfinite(sub['E_theta_mean'])
            ax.plot(sub.loc[valid, 'kappa_M'], sub.loc[valid, 'E_theta_mean'], 'o-', label=f'epsilon={eps:g}')
        ax.set(xscale='log', yscale='log', xlabel='Prescribed condition number of M', ylabel='Mean relative parameter error', title='Direct perturbation: conditioning amplification')
        ax.grid(True, which='both', alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(out / 'perturbation_amplification.png', dpi=250)
        plt.close(fig)
        print('Perturbation analysis:', out)
    else:
        print('No perturbation aggregate found; skipping perturbation analysis.')
    print('Analysis complete:', out)

if __name__ == '__main__':
    main()
