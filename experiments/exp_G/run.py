import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulation import DistributedRegression, THETA_ERROR_EPSILON

SLOPES = (np.array([1.0, 0.5, 0.0]), np.array([-0.5, 0.2, 5.0]))


def dataset_file(seed, agents, observations, noise, profile="thesis"):
    directory = HERE / 'data'
    if profile == 'thesis':
        directory = directory / 'contiguous'
    return directory / f'Simpson_A{agents}_n{observations}_noise{noise:g}_seed{seed}.csv'


def generate_dataset(path, seed, agents, observations, noise, profile="thesis"):
    rng = np.random.RandomState(seed)
    M = np.empty((agents, 3, 3), dtype=float)
    B = np.empty((agents, 3), dtype=float)
    n = np.full(agents, observations, dtype=float)
    for agent in range(agents):
        group = int(agent >= agents // 2) if profile == "thesis" else agent % 2
        X = rng.normal(size=(observations, 2))
        X_aug = np.column_stack((X, np.ones(observations)))
        y = X_aug @ SLOPES[group] + rng.normal(scale=noise, size=observations)
        M[agent] = (X_aug.T @ X_aug) / observations
        B[agent] = (X_aug.T @ y) / observations
    data = {'agent': np.arange(agents), 'n': n}
    for i in range(3):
        for j in range(3):
            data[f'M_{i}_{j}'] = M[:, i, j]
    for i in range(3):
        data[f'B_{i}'] = B[:, i]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(data).to_csv(path, index=False, float_format='%.17g')
    return M, B, n


def load_dataset(path):
    df = pd.read_csv(path)
    M = np.stack([df[f'M_{i}_{j}'].to_numpy(float) for i in range(3)
                  for j in range(3)], axis=1).reshape(-1, 3, 3)
    B = np.column_stack([df[f'B_{i}'].to_numpy(float) for i in range(3)])
    n = df['n'].to_numpy(float)
    if len(M) < 2 or not np.isfinite(M).all() or not np.isfinite(B).all() or not np.isfinite(n).all() or np.any(n <= 0):
        raise ValueError(f'Invalid sufficient-statistics CSV: {path}')
    return M, B, n


def relative_error(theta, target):
    if not np.isfinite(theta).all():
        return float('nan')
    return float(np.linalg.norm(theta - target) / max(np.linalg.norm(target), THETA_ERROR_EPSILON))


def run_seed(seed, csv, steps, record_every, progress_every, cond_threshold, outdir, profile="thesis"):
    M, B, n = load_dataset(csv)
    N, p = B.shape
    theta_global = np.linalg.lstsq(np.mean(M, axis=0), np.mean(B, axis=0), rcond=None)[0]
    local = []
    for m, b in zip(M, B):
        try:
            local.append(np.linalg.solve(m, b))
        except np.linalg.LinAlgError:
            local.append(np.full(p, np.nan))
    local = np.asarray(local)
    valid = np.isfinite(local).all(axis=1)
    unweighted = np.mean(local[valid], axis=0) if valid.any() else np.full(p, np.nan)
    weighted = np.average(local[valid], weights=n[valid], axis=0) if valid.any() else np.full(p, np.nan)

    sim = DistributedRegression(
        X_seen_aug=np.ones((N, p)), y_seen=np.ones(N),
        X_global_aug=np.ones((N, p)), y_global=np.ones(N),
        cond_threshold=cond_threshold, gossip_mode='coordinate',
        stop_on_theta_change=(profile == "thesis"), epsilon=1e-10, r=20)
    sim.agents_M = M.copy()
    sim.agents_B = B.copy()
    sim.M_global = M.mean(axis=0)
    sim.B_global = B.mean(axis=0)
    sim.theta_global = theta_global
    sim.theta_seen = theta_global
    sim.w_initial_sum = np.concatenate([M.reshape(N, -1), B], axis=1).sum(axis=0)

    history = []
    start = time.perf_counter()
    for step in range(1, steps + 1):
        sim.gossip_step()
        stopped = sim.stop_on_theta_change and sim.all_inactive()
        endpoint = stopped or step == steps
        record = endpoint or step % record_every == 0
        report = progress_every and (endpoint or step % progress_every == 0)
        if record or report:
            thetas, conds, stable = sim.compute_theta_agents()
            good = thetas[stable]
            err = float(sim.relative_theta_error(good))
            fraction = float(np.mean(stable))
            if record:
                history.append({'step': step, 'theta_error': err,
                                'cond_mean': float(np.mean(conds)),
                                'cond_max': float(np.max(conds)),
                                'stable_fraction': fraction})
            if report:
                print(f'  Step {step:,}/{steps:,} ({100*step/steps:.1f}%) | '
                      f'E_theta={err:.3e} | stable={fraction:.3f} | '
                      f'elapsed={time.perf_counter()-start:.1f}s', flush=True)

        if endpoint:
            break

    final_mean = good.mean(axis=0) if len(good) else np.full(p, np.nan)
    pd.DataFrame(history).to_csv(outdir / f'convergence_seed{seed}.csv', index=False)
    return {'seed': seed, 'theta_global': theta_global.tolist(),
            'unweighted_avg': unweighted.tolist(), 'weighted_avg': weighted.tolist(),
            'err_unweighted': relative_error(unweighted, theta_global),
            'err_weighted': relative_error(weighted, theta_global),
            'gossip_final_err': relative_error(final_mean, theta_global),
            'gossip_final_theta': final_mean.tolist(),
            'gossip_final_mean_agent_error': err, 'profile': profile,
            'steps_run': step, 'step_budget': steps,
            'stop_reason': 'all_inactive' if stopped else 'step_budget',
            'stable_fraction': fraction}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=['thesis', 'archive'], default='thesis',
                        help='Thesis uses contiguous groups and heuristic stopping; archive uses alternating groups and a fixed budget.')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 21)))
    parser.add_argument('--agents', type=int, default=100)
    parser.add_argument('--observations_per_agent', type=int, default=10)
    parser.add_argument('--noise', type=float, default=0.1)
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--record_every', type=int, default=100)
    parser.add_argument('--progress_every', type=int, default=5000)
    parser.add_argument('--cond_threshold', type=float, default=50)
    parser.add_argument('--generate_only', action='store_true')
    parser.add_argument('--regenerate', action='store_true', help='Overwrite existing dataset CSVs')
    args = parser.parse_args()
    if args.agents < 2 or args.observations_per_agent < 1 or args.steps < 1 or args.record_every < 1 or args.progress_every < 0:
        parser.error('Invalid nonpositive experiment settings')
    outdir = HERE / 'results' / f'exp_G_{datetime.now():%Y%m%d_%H%M%S_%f}'
    if not args.generate_only:
        outdir.mkdir(parents=True)
    results = []
    for index, seed in enumerate(args.seeds, 1):
        csv = dataset_file(seed, args.agents, args.observations_per_agent, args.noise, args.profile)
        if args.regenerate or not csv.is_file():
            generate_dataset(csv, seed, args.agents, args.observations_per_agent, args.noise, args.profile)
            print(f'Generated {csv.name}', flush=True)
        if args.generate_only:
            continue
        print(f'\n[{index}/{len(args.seeds)}] Seed {seed} | {args.steps:,} steps | {csv.name}', flush=True)
        np.random.seed(seed)
        result = run_seed(seed, csv, args.steps, args.record_every,
                          args.progress_every, args.cond_threshold, outdir, args.profile)
        results.append(result)
        (outdir / 'results.json').write_text(json.dumps(results, indent=2))
        print(f"  Done | steps={result['steps_run']} | mean agent error={result['gossip_final_mean_agent_error']:.3e} | "
              f"unweighted={result['err_unweighted']:.3e} | "
              f"weighted={result['err_weighted']:.3e}", flush=True)
    print(f'\nDatasets: {HERE / "data"}', flush=True)
    if not args.generate_only:
        print(f'Completed: {outdir}', flush=True)


if __name__ == '__main__':
    main()
