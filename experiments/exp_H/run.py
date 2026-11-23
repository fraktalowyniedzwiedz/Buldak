"""California Housing
"""
import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulation import DistributedRegression


def dataset_file(seed, fraction):
    return HERE / "data" / f"CA_Seen{round(fraction*100):03d}_seed{seed}.csv"


def prepare_dataset(seed, fraction, force=False):
    path = dataset_file(seed, fraction)
    if path.exists() and not force:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    housing = fetch_california_housing()
    X = np.asarray(housing.data, dtype=float)
    y = np.asarray(housing.target, dtype=float)
    rng = np.random.default_rng(seed)
    permutation = rng.permutation(len(y))
    n_seen = min(len(y), max(1, int(round(len(y) * fraction))))
    seen = np.zeros(len(y), dtype=bool)
    seen[permutation[:n_seen]] = True
    scaler = StandardScaler().fit(X[seen])
    X = scaler.transform(X)
    df = pd.DataFrame(X, columns=[f"x{i}" for i in range(X.shape[1])])
    df["y"] = y
    df["split"] = np.where(seen, "seen", "unseen")
    df.to_csv(path, index=False)
    return path


def regression_metrics(X, y, theta):
    if len(y) == 0:
        return None
    pred = X @ theta
    return {"RMSE": float(np.sqrt(mean_squared_error(y, pred))),
            "MAE": float(mean_absolute_error(y, pred)),
            "R2": float(r2_score(y, pred)) if len(y) > 1 else None}


def simulate(csv, outdir, seed, mode, steps, progress_every, checkpoint_every, cond_threshold):
    np.random.seed(seed)
    df = pd.read_csv(csv)
    features = [c for c in df.columns if c.startswith("x") and c[1:].isdigit()]
    features.sort(key=lambda x: int(x[1:]))
    X = df[features].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    seen = df["split"].eq("seen").to_numpy()
    X_aug = np.column_stack((X, np.ones(len(X))))
    sim = DistributedRegression(
        X_seen_aug=X_aug[seen], y_seen=y[seen],
        X_global_aug=X_aug, y_global=y,
        cond_threshold=cond_threshold, gossip_mode=mode,
        stop_on_theta_change=False, epsilon=1e-10, r=20,
    )
    outdir.mkdir(parents=True, exist_ok=True)
    rows = []
    started = time.monotonic()
    for step in range(1, steps + 1):
        sim.gossip_step()
        if step == 1 or step % checkpoint_every == 0 or step == steps:
            theta, conds, stable = sim.compute_theta_agents()
            M_diff = sim.agents_M - sim.M_global
            B_diff = sim.agents_B - sim.B_global
            w = np.concatenate((sim.agents_M.reshape(sim.N, -1), sim.agents_B), axis=1)
            point = {
                "step": step,
                "scalars_transmitted": step * 2 * sim.q,
                "usable_agents": int(stable.sum()),
                "theta_error": float(sim.relative_theta_error(theta)),
                "E_M": float(np.mean(np.sum(M_diff*M_diff, axis=(1,2))) /
                             (np.linalg.norm(sim.M_global, "fro")**2 + 1e-12)),
                "E_B": float(np.mean(np.sum(B_diff*B_diff, axis=1)) /
                             (np.linalg.norm(sim.B_global)**2 + 1e-12)),
                "mass_drift": float(np.linalg.norm(w.sum(axis=0) - sim.w_initial_sum)),
                "cond_mean": float(np.mean(conds)),
                "cond_max": float(np.max(conds)),
                "stable_fraction": float(np.mean(stable)),
            }
            rows.append(point)
            if step == 1 or step % progress_every == 0 or step == steps:
                print(f"  step {step:,}/{steps:,} | E_theta={point['theta_error']:.3e} "
                      f"| E_M={point['E_M']:.3e} | E_B={point['E_B']:.3e} "
                      f"| stable={point['stable_fraction']:.1%} "
                      f"| elapsed={time.monotonic()-started:.1f}s", flush=True)
    pd.DataFrame(rows).to_csv(outdir / "convergence.csv", index=False)
    metrics = {"steps": steps, "final_stable_fraction": float(np.mean(stable)),
               "rel_error_seen": None, "rel_error_global": None,
               "Seen": None, "Unseen": None}
    metrics.update(
        seed=seed, mode=mode, seen_fraction=float(seen.mean()),
        n_seen=int(seen.sum()), n_unseen=int((~seen).sum()),
        usable_agents=int(stable.sum()), cond_threshold=cond_threshold,
        checkpoint_every=checkpoint_every, width=sim.q, target_unit_dollars=100000,
        predictor="mean of usable agent coefficient vectors",
        centralized={
            "Seen": regression_metrics(X_aug[seen], y[seen], sim.theta_seen),
            "Unseen": regression_metrics(X_aug[~seen], y[~seen], sim.theta_seen),
        },
    )
    if np.any(stable):
        avg = np.mean(theta[stable], axis=0)
        metrics["rel_error_seen"] = float(
            np.linalg.norm(avg - sim.theta_seen)/(np.linalg.norm(sim.theta_seen)+1e-12))
        metrics["rel_error_global"] = float(
            np.linalg.norm(avg - sim.theta_global)/(np.linalg.norm(sim.theta_global)+1e-12))
        metrics["Seen"] = regression_metrics(X_aug[seen], y[seen], avg)
        metrics["Unseen"] = regression_metrics(X_aug[~seen], y[~seen], avg)
    (outdir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  completed in {time.monotonic()-started:.1f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seen_fractions", type=float, nargs="+", default=[.10, .25, .50, 1.0])
    parser.add_argument("--modes", nargs="+", choices=["full", "coordinate"],
                        default=["full", "coordinate"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1,2,3])
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--progress_every", type=int, default=5000)
    parser.add_argument("--checkpoint_every", type=int, default=100)
    parser.add_argument("--cond_threshold", type=float, default=50)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if any(not 0 < f <= 1 for f in args.seen_fractions):
        parser.error("seen fractions must be in (0,1]")
    if args.steps < 1 or args.progress_every < 1 or args.checkpoint_every < 1:
        parser.error("steps and reporting intervals must be positive")
    for fraction in args.seen_fractions:
        for seed in args.seeds:
            prepare_dataset(seed, fraction, args.regenerate)
    print("Prepared CSV datasets in", HERE / "data", flush=True)
    if args.generate_only:
        return
    output = HERE / "results" / ("exp_H_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    total = len(args.seen_fractions)*len(args.modes)*len(args.seeds)
    i = 0
    for fraction in args.seen_fractions:
        for mode in args.modes:
            for seed in args.seeds:
                i += 1
                tag = "full" if mode == "full" else "coord"
                name = f"CA_Seen{round(fraction*100):03d}_Mode{tag}_seed{seed}"
                print(f"\n[{i}/{total}] {name}", flush=True)
                simulate(dataset_file(seed, fraction), output / name, seed, mode,
                         args.steps, args.progress_every, args.checkpoint_every,
                         args.cond_threshold)
    print("Completed:", output, flush=True)


if __name__ == "__main__":
    main()
