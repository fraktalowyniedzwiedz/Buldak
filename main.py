import argparse
import sys
from pathlib import Path
import json
import os
import time
import pandas as pd

from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from simulation import DistributedRegression
from plotting import (
    plot_theta_error,
    plot_condition_numbers,
    plot_matrix_history,
    plot_theta_history,
    plot_regression_metrics,
    plot_error_distribution,
    plot_norm_history,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Run distributed regression simulation.")
    parser.add_argument("--data_file", type=str, required=True,
                        help="Pre-generated CSV dataset file (x0, x1, ..., y columns)")
    # simulation params
    parser.add_argument("--seen_fraction", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--steps", type=int, default=500000)
    parser.add_argument("--cond_threshold", type=float, default=50)
    parser.add_argument("--stop_on_theta_change", action="store_true", default=True)
    parser.add_argument("--epsilon", type=float, default=1e-10)
    parser.add_argument("--r", type=int, default=20)
    parser.add_argument("--gossip_mode", type=str, default="full",
                        choices=["full", "coordinate"],
                        help="full: average all coordinates; coordinate: average a uniform q-subset")
    parser.add_argument("--q", type=int, default=1,
                        help="Block width for coordinate mode (1 <= q <= p*(p+1)/2+p); full mode always averages all coordinates")
    # output
    parser.add_argument("--output_dir", type=str, default="results",
                        help="Base directory for saving results")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="Subdirectory name inside output_dir")
    parser.add_argument(
        "--stop_when_done",
        action="store_true",
        default=False,
        help="Accepted for compatibility. The simulation always stops when all agents are inactive."
    )
    parser.add_argument(
        "--run_until_convergence",
        action="store_true",
        default=False,
        help="Ignore --steps and run until all agents are inactive (no maximum)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    np.random.seed(args.seed)

    data_path = Path(args.data_file).expanduser().resolve()
    if not data_path.is_file():
        raise FileNotFoundError(f"Pre-generated CSV dataset not found: {data_path}. Run generate_data.py first.")
    data = pd.read_csv(data_path)
    if "y" not in data.columns:
        raise ValueError(f"Dataset CSV must contain a 'y' target column: {data_path}")
    feature_columns = [column for column in data.columns if column != "y"]
    X = data[feature_columns].to_numpy(dtype=float)
    y = data["y"].to_numpy(dtype=float)
    X_aug = np.hstack((X, np.ones((len(X), 1))))
    class LoadedDataset:
        pass
    dataset = LoadedDataset()
    dataset.name = "synthetic"
    dataset.X_seen = X
    dataset.y_seen = y
    dataset.X_unseen = np.empty((0, X.shape[1]))
    dataset.y_unseen = np.empty(0)
    dataset.X_all = X
    dataset.y_all = y
    dataset.X_seen_aug = X_aug
    dataset.X_unseen_aug = np.empty((0, X.shape[1] + 1))
    dataset.X_all_aug = X_aug
    dataset.dim = X.shape[1]

    if args.experiment_name is None:
        exp_name = (
            f"{dataset.name}"
            f"_Seen{len(dataset.y_seen)}"
            f"_D{dataset.dim}"
            f"_seed{args.seed}"
        )
    else:
        exp_name = args.experiment_name

    outdir = os.path.join(args.output_dir, exp_name)
    os.makedirs(outdir, exist_ok=True)

    sim = DistributedRegression(
        X_seen_aug=dataset.X_seen_aug,
        y_seen=dataset.y_seen,
        X_global_aug=dataset.X_all_aug,
        y_global=dataset.y_all,
        cond_threshold=args.cond_threshold,
        gossip_mode=args.gossip_mode,
        stop_on_theta_change=args.stop_on_theta_change,
        epsilon=args.epsilon,
        r=args.r,
        q=args.q,
    )

    

    print(f"\n=== Experiment: {exp_name} ===")
    print(f"Global theta:\n{sim.theta_global}")
    print(f"Seen theta:\n{sim.theta_seen}")

    theta_error_history = []
    theta_error_steps = []
    cond_mean_history = []
    cond_max_history = []
    stable_fraction_history = []
    condition_steps = []
    matrix_history = []
    theta_history = []
    steps_history = []
    l1_history = []
    l2_history = []
    linf_history = []
    norm_steps = []
    E_M_history = []
    E_B_history = []
    mass_drift_history = []

    start = time.perf_counter()

    step = 0

    while True:
        result = sim.gossip_step()

        if result is not None:
            l1, l2, linf = result
            l1_history.append(l1)
            l2_history.append(l2)
            linf_history.append(linf)
            norm_steps.append(step)

        all_finished = sim.all_inactive()
        budget_reached = not args.run_until_convergence and step + 1 >= args.steps
        if step % 100 == 0 or all_finished or budget_reached:
            theta_agents, conds, stable = sim.compute_theta_agents()

            # parameter error
            theta_err = sim.relative_theta_error(theta_agents)
            theta_error_history.append(theta_err)
            theta_error_steps.append(step)

            # condition numbers
            cond_mean_history.append(np.mean(conds))
            cond_max_history.append(np.max(conds))
            stable_fraction_history.append(np.mean(stable))
            condition_steps.append(step)

            # statistic errors and mass drift
            current_M_flat = sim.agents_M.reshape(sim.N, -1)
            current_B = sim.agents_B
            current_w_flat = np.concatenate([current_M_flat, current_B], axis=1)
            mass_drift = np.linalg.norm(current_w_flat.sum(axis=0) - sim.w_initial_sum)

            # E_M: relative Frobenius error (average over agents)
            M_diff = sim.agents_M - sim.M_global
            E_M = np.mean([np.linalg.norm(m, ord='fro')**2 for m in M_diff]) / (np.linalg.norm(sim.M_global, ord='fro')**2 + 1e-12)

            # E_B: relative L2 error (average over agents)
            B_diff = sim.agents_B - sim.B_global
            E_B = np.mean([np.linalg.norm(b)**2 for b in B_diff]) / (np.linalg.norm(sim.B_global)**2 + 1e-12)

            E_M_history.append(E_M)
            E_B_history.append(E_B)
            mass_drift_history.append(mass_drift)

            stable_theta = theta_agents[stable]
            stable_M = sim.agents_M[stable]
            if len(stable_theta) > 0:
                steps_history.append(step)
                matrix_history.append(stable_M.mean(axis=0))
                theta_history.append(stable_theta.mean(axis=0))

            if step % 5000 == 0:
                elapsed = time.perf_counter() - start
                print(
                    f"[{step}/{args.steps}] "
                    f"Elapsed={elapsed:.2f}s "
                    f"ThetaErr={theta_error_history[-1]:.3e} "
                    f"Stable={stable_fraction_history[-1]:.3f}"
                )
        step += 1
        if all_finished:
            print(f"All agents finished after {step} interactions.")
            break
        if budget_reached:
            break

    convergence_df = pd.DataFrame({
        'step': theta_error_steps,
        'interactions': np.asarray(theta_error_steps) + 1,
        'theta_error': theta_error_history,
        'E_M': E_M_history,
        'E_B': E_B_history,
        'mass_drift': mass_drift_history,
        'cond_mean': cond_mean_history,
        'cond_max': cond_max_history,
        'stable_fraction': stable_fraction_history,
    })
    convergence_df.to_csv(os.path.join(outdir, 'convergence.csv'), index=False)

    # final theta
    theta_agents, conds, stable = sim.compute_theta_agents()
    stable_theta = theta_agents[stable]
    print("\nFINAL RESULTS")
    print(f"Global theta:\n{sim.theta_global}")

    metrics = {
        "seed": args.seed,
        "steps": args.steps,
        "interactions_completed": step,
        "stop_reason": "all_inactive" if all_finished else "step_budget",
        "final_stable_fraction": float(np.mean(stable)),
    }
    if len(stable_theta) > 0:
        mean_theta = stable_theta.mean(axis=0)

        # predictions
        seen_pred = dataset.X_seen_aug @ mean_theta
        unseen_pred = dataset.X_unseen_aug @ mean_theta

        seen_errors = np.abs(dataset.y_seen - seen_pred)
        unseen_errors = np.abs(dataset.y_unseen - unseen_pred)

        rel_error_seen = (
            np.linalg.norm(mean_theta - sim.theta_seen) /
            np.linalg.norm(sim.theta_seen)
        )
        rel_error_global = (
            np.linalg.norm(mean_theta - sim.theta_global) /
            np.linalg.norm(sim.theta_global)
        )

        print(f"Mean stable theta:\n{mean_theta}")
        print(f"Relative error to seen: {rel_error_seen:.6e}")
        print(f"Relative error to global: {rel_error_global:.6e}")

        # regression metrics on seen and unseen (if they exist)
        def evaluate(X_aug, y, theta):
            y_pred = X_aug @ theta
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            mae = mean_absolute_error(y, y_pred)
            r2 = r2_score(y, y_pred)
            return rmse, mae, r2

        if len(dataset.y_seen) > 0:
            seen_rmse, seen_mae, seen_r2 = evaluate(
                dataset.X_seen_aug, dataset.y_seen, mean_theta
            )
            metrics["Seen"] = {"RMSE": seen_rmse, "MAE": seen_mae, "R2": seen_r2}
            print(f"\nSEEN: RMSE={seen_rmse:.6f}, MAE={seen_mae:.6f}, R2={seen_r2:.6f}")

        if len(dataset.y_unseen) > 0:
            unseen_rmse, unseen_mae, unseen_r2 = evaluate(
                dataset.X_unseen_aug, dataset.y_unseen, mean_theta
            )
            metrics["Unseen"] = {"RMSE": unseen_rmse, "MAE": unseen_mae, "R2": unseen_r2}
            print(f"\nUNSEEN: RMSE={unseen_rmse:.6f}, MAE={unseen_mae:.6f}, R2={unseen_r2:.6f}")

        # Save metrics
        metrics.update({
            "rel_error_seen": rel_error_seen,
            "rel_error_global": rel_error_global,
        })

        plot_regression_metrics(metrics, exp_name, base_dir=args.output_dir)
        if len(dataset.y_seen) > 0 and len(dataset.y_unseen) > 0:
            plot_error_distribution(seen_errors, unseen_errors, exp_name, base_dir=args.output_dir)

    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # convergence plots
    plot_theta_error(theta_error_history, theta_error_steps, exp_name, base_dir=args.output_dir)
    plot_condition_numbers(
        cond_mean_history, cond_max_history, stable_fraction_history,
        condition_steps, exp_name, base_dir=args.output_dir
    )
    if len(matrix_history) > 0:
        plot_matrix_history(matrix_history, steps_history, exp_name, base_dir=args.output_dir)
        plot_theta_history(theta_history, steps_history, sim.theta_seen, exp_name, base_dir=args.output_dir)
    if len(norm_steps) > 0:
        plot_norm_history(l1_history, l2_history, linf_history, norm_steps, exp_name, base_dir=args.output_dir)

    print("\nSimulation complete.")


if __name__ == "__main__":
    main()