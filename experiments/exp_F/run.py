import argparse
from datetime import datetime
from pathlib import Path
import sys
import json
import time

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from simulation import DistributedRegression

import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
NOISES = [0.0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]


def simulate(csv, outdir, seed, steps, progress_every=10000):
    np.random.seed(seed)
    frame = pd.read_csv(csv)
    features = [c for c in frame if c not in ('y', 'split')]
    X = frame[features].to_numpy(dtype=float)
    y = frame['y'].to_numpy(dtype=float)
    seen = frame['split'].eq('seen').to_numpy()
    if not seen.any() or not (~seen).any():
        raise ValueError('exp_F requires nonempty training and test partitions')
    X_aug = np.column_stack((X, np.ones(len(X))))
    sim = DistributedRegression(
        X_seen_aug=X_aug[seen], y_seen=y[seen],
        X_global_aug=X_aug, y_global=y,
        cond_threshold=50, gossip_mode='coordinate',
        stop_on_theta_change=False, epsilon=1e-10, r=20,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    history = []
    started = time.monotonic()
    for step in range(steps):
        sim.gossip_step()
        if step % 100 == 0:
            theta, conds, stable = sim.compute_theta_agents()
            E_M = np.mean(np.sum((sim.agents_M - sim.M_global)**2, axis=(1, 2))) / (np.linalg.norm(sim.M_global, 'fro')**2 + 1e-12)
            E_B = np.mean(np.sum((sim.agents_B - sim.B_global)**2, axis=1)) / (np.linalg.norm(sim.B_global)**2 + 1e-12)
            history.append({
                'step': step, 'theta_error': sim.relative_theta_error(theta),
                'E_M': E_M, 'E_B': E_B,
                'stable_fraction': float(np.mean(stable)),
                'cond_mean': float(np.mean(conds)), 'cond_max': float(np.max(conds)),
            })
            if (step + 1) % progress_every == 0 or step == 0:
                point = history[-1]
                print(f"  step {step + 1:,}/{steps:,} ({(step + 1) / steps:.0%}) | "
                      f"E_theta={point['theta_error']:.4g} | "
                      f"E_M={point['E_M']:.4g} | E_B={point['E_B']:.4g} | "
                      f"stable={point['stable_fraction']:.0%} | "
                      f"elapsed={time.monotonic() - started:.1f}s", flush=True)
    pd.DataFrame(history).to_csv(outdir / 'convergence.csv', index=False)
    theta, _, stable = sim.compute_theta_agents()
    metrics = {'steps': steps, 'final_stable_fraction': float(np.mean(stable))}
    if np.any(stable):
        mean_theta = theta[stable].mean(axis=0)
        for name, mask in [('Seen', seen), ('Unseen', ~seen)]:
            truth = y[mask]
            pred = X_aug[mask] @ mean_theta
            metrics[name] = {
                'RMSE': float(np.sqrt(mean_squared_error(truth, pred))),
                'MAE': float(mean_absolute_error(truth, pred)),
                'R2': float(r2_score(truth, pred)),
            }
        metrics['rel_error_seen'] = float(np.linalg.norm(mean_theta - sim.theta_seen) / (np.linalg.norm(sim.theta_seen) + 1e-12))
        metrics['rel_error_global'] = float(np.linalg.norm(mean_theta - sim.theta_global) / (np.linalg.norm(sim.theta_global) + 1e-12))
    else:
        metrics['rel_error_seen'] = None
        metrics['rel_error_global'] = None
    with (outdir / 'metrics.json').open('w') as f:
        json.dump(metrics, f, indent=2)
    print(f"  Finished in {time.monotonic() - started:.1f}s | "
          f"stable={metrics['final_stable_fraction']:.0%} | "
          f"test RMSE={metrics.get('Unseen', {}).get('RMSE', float('nan')):.4g}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--A", type=int, default=100)
    parser.add_argument("--D", type=int, default=2)
    parser.add_argument("--seen_fraction", type=float, default=0.8)
    parser.add_argument("--noises", type=float, nargs="+", default=NOISES)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--steps", type=int, default=200000)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--progress_every", type=int, default=10000,
                        help="Print progress every N gossip steps (default: 10000)")
    args = parser.parse_args()
    if args.progress_every < 1:
        parser.error("--progress_every must be positive")
    if not 0 < args.seen_fraction < 1:
        parser.error("--seen_fraction must be in (0, 1) for held-out evaluation")
    data_dir = HERE / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    output = HERE / "results" / ("exp_F_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    if not args.generate_only:
        output.mkdir(parents=True, exist_ok=True)
    total = len(args.noises) * len(args.seeds)
    run_number = 0
    for noise in args.noises:
        for seed in args.seeds:
            run_number += 1
            X, y = make_regression(n_samples=args.A, n_features=args.D,
                                   noise=noise, random_state=seed)
            indices = np.arange(args.A)
            seen_idx, unseen_idx = train_test_split(
                indices, train_size=args.seen_fraction, random_state=seed)
            scaler = StandardScaler().fit(X[seen_idx])
            X = scaler.transform(X)
            frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(args.D)])
            frame["y"] = y
            frame["split"] = "unseen"
            frame.loc[seen_idx, "split"] = "seen"
            filename = f"NoiseSep_{noise}_A{args.A}_D{args.D}_seed{seed}"
            csv = data_dir / f"{filename}.csv"
            frame.to_csv(csv, index=False)
            if args.generate_only:
                continue
            print(f"\n[{run_number}/{total}] {filename} | {args.steps:,} steps", flush=True)
            simulate(csv, output / filename, seed, args.steps, args.progress_every)
    print("Datasets generated in:", data_dir)
    if not args.generate_only:
        print("Results saved in:", output)


if __name__ == "__main__":
    main()
