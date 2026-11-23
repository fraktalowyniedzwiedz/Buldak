import argparse
from datetime import datetime
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parents[1]

def make_problem(A, p, kappa_M, rng):
    """Z (A x p) with sigma_min(Z) = 1/sqrt(kappa_M), sigma_max(Z) = 1"""
    s = np.ones(p)
    s[-1] = 1.0 / np.sqrt(kappa_M)

    U_full, _ = np.linalg.qr(rng.standard_normal((A, p)))
    V, _ = np.linalg.qr(rng.standard_normal((p, p)))
    Z = U_full @ np.diag(s) @ V.T
    theta_star_true = rng.standard_normal(p)
    y = Z @ theta_star_true
    return Z, y

def run_one(A, p, kappa_M, eps_M, eps_B, seed):
    rng = np.random.default_rng(seed)
    Z, y = make_problem(A, p, kappa_M, rng)

    M_bar = Z.T @ Z / A
    B_bar = Z.T @ y / A
    theta_hat = np.linalg.solve(M_bar, B_bar)

    M_norm = np.linalg.norm(M_bar, 2)
    B_norm = np.linalg.norm(B_bar, 2)
    Minv_norm = 1.0 / np.linalg.svd(M_bar, compute_uv=False)[-1]

    thetas = np.zeros((A, p))
    DeltaM_2 = np.zeros(A)
    DeltaB_2 = np.zeros(A)
    bounds = np.zeros(A)
    valid = np.ones(A, dtype=bool)

    for i in range(A):
        dM = rng.standard_normal((p, p))
        dM = 0.5 * (dM + dM.T) # symmetric
        dM *= eps_M * M_norm / np.linalg.norm(dM, 'fro')
        dB = rng.standard_normal(p)
        dB *= eps_B * B_norm / np.linalg.norm(dB)

        M_i, B_i = M_bar + dM, B_bar + dB
        DeltaM_2[i] = np.linalg.norm(dM, 2)
        DeltaB_2[i] = np.linalg.norm(dB, 2)

        rho = Minv_norm * DeltaM_2[i]
        if rho >= 1.0:
            valid[i] = False
            continue

        thetas[i] = np.linalg.solve(M_i, B_i)
        bounds[i] = (Minv_norm / (1.0 - rho)) * (
            DeltaB_2[i] + DeltaM_2[i] * np.linalg.norm(theta_hat, 2)
        )

    denom = max(np.linalg.norm(theta_hat, 2), 1e-12)
    errs = np.linalg.norm(thetas - theta_hat, axis=1) / denom

    # Bound in the same relative units
    bounds_rel = bounds / denom

    errs_v, bounds_v = errs[valid], bounds_rel[valid]

    return dict(
        A=A, p=p, kappa_M=kappa_M, eps_M=eps_M, eps_B=eps_B, seed=seed,
        E_theta_mean=float(errs_v.mean()) if len(errs_v) else np.nan,
        E_theta_median=float(np.median(errs_v)) if len(errs_v) else np.nan,
        E_theta_max=float(errs_v.max()) if len(errs_v) else np.nan,
        bound_mean=float(bounds_v.mean()) if len(bounds_v) else np.nan,
        bound_max=float(bounds_v.max()) if len(bounds_v) else np.nan,
        stable_fraction=float(valid.mean()),
        n_valid=int(valid.sum()),
    )

def run_perturbation(out_dir, A, D, kappas, seeds):
    out_dir.mkdir(parents=True, exist_ok=True)
    eps_pairs = [(1e-3, 1e-3), (1e-4, 1e-4), (1e-5, 1e-5), (1e-6, 1e-6)]
    records = [run_one(A, D + 1, k, em, eb, seed)
               for k in kappas for em, eb in eps_pairs for seed in seeds]
    df = pd.DataFrame(records)
    df.to_csv(out_dir / 'perturbation_runs.csv', index=False)
    agg = df.groupby(['kappa_M', 'eps_M']).agg(
        E_theta_mean=('E_theta_mean', 'mean'),
        E_theta_std=('E_theta_mean', 'std'),
        E_theta_median=('E_theta_median', 'mean'),
        bound_mean=('bound_mean', 'mean'),
        bound_std=('bound_mean', 'std'),
        stable_fraction=('stable_fraction', 'mean'),
    ).reset_index()
    agg.to_csv(out_dir / 'perturbation_aggregate.csv', index=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output_dir', type=Path)
    p.add_argument('--A', type=int, default=100)
    p.add_argument('--D', type=int, default=5)
    p.add_argument('--noise', type=float, default=0.1)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--kappas', type=float, nargs='+', default=[1, 10, 100, 1000, 10000, 100000, 1000000])
    p.add_argument('--steps', type=int, default=200000)
    p.add_argument('--only', choices=['all', 'gossip', 'perturbation'], default='all')
    args = p.parse_args()
    root = (args.output_dir or HERE / 'results' / ('exp_E_' + datetime.now().strftime('%Y%m%d_%H%M%S'))).resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.only in ('all', 'gossip'):
        subprocess.run([
            sys.executable, str(HERE / 'generate_data.py'),
            '--A', str(args.A), '--D', str(args.D), '--noise', str(args.noise),
            '--seeds', *map(str, args.seeds), '--kappas', *map(str, args.kappas)
        ], check=True)
        out = root / 'gossip'
        out.mkdir(exist_ok=True)
        for kappa in args.kappas:
            for seed in args.seeds:
                name = f'Amp_kappa{kappa:g}_A{args.A}_D{args.D}_seed{seed}'
                data = HERE / 'data' / f'A{args.A}_D{args.D}_kappa{kappa:g}_seed{seed}.csv'
                subprocess.run([
                    sys.executable, str(HERE / 'run_with_spectral_metrics.py'),
                    '--data_file', str(data), '--seen_fraction', '1.0',
                    '--seed', str(seed), '--steps', str(args.steps),
                    '--gossip_mode', 'coordinate', '--epsilon', '1e-30',
                    '--r', '20', '--cond_threshold', '1e12',
                    '--output_dir', str(out), '--experiment_name', name
                ], check=True, cwd=PROJECT_ROOT)
    if args.only in ('all', 'perturbation'):
        run_perturbation(root / 'perturbation', args.A, args.D, args.kappas, range(1, 21))
    print('Completed:', root)


if __name__ == '__main__':
    main()
