import os
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt

RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path):
    os.makedirs(
        path,
        exist_ok=True,
    )

def plot_error_distribution(
    seen_errors,
    unseen_errors,
    experiment_name,
    base_dir="results",
):
    outdir = os.path.join(base_dir, experiment_name)
    ensure_dir(outdir)

    plt.figure(figsize=(8, 5))

    plt.hist(
        seen_errors,
        bins=40,
        density=True,
        alpha=0.6,
        label="Seen",
    )

    plt.hist(
        unseen_errors,
        bins=40,
        density=True,
        alpha=0.6,
        label="Unseen",
    )

    plt.xlabel(
        "Absolute prediction error"
    )

    plt.ylabel("Density")

    plt.title(
        "Prediction error distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, f"error_distribution_{RUN_TIMESTAMP}.png"),
        dpi=300,
    )

    plt.close()

def plot_theta_error(
    theta_error_history,
    theta_error_steps,
    experiment_name,
    base_dir="results",
):
    outdir = os.path.join(base_dir, experiment_name)
    ensure_dir(outdir)

    plt.figure(figsize=(8, 5))

    plt.plot(
        theta_error_steps,
        theta_error_history,
    )

    plt.yscale("log")

    plt.xlabel("Gossip Step")

    plt.ylabel(
        "Relative Theta Error"
    )

    plt.title(
        "Theta Error"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, f"theta_error_{RUN_TIMESTAMP}.png"),
        dpi=300,
    )

    plt.close()

def plot_condition_numbers(
    cond_mean_history,
    cond_max_history,
    stable_fraction_history,
    condition_steps,
    experiment_name,
    base_dir="results",
):
    outdir = os.path.join(base_dir, experiment_name)
    ensure_dir(outdir)

    # condition numbers full run, logarithmic

    plt.figure(figsize=(8, 5))

    plt.plot(
        condition_steps,
        cond_mean_history,
        label="Mean Cond",
    )

    plt.plot(
        condition_steps,
        cond_max_history,
        label="Max Cond",
    )

    plt.yscale("log")

    plt.legend()

    plt.xlabel("Gossip Step")

    plt.ylabel(
        "Condition Number"
    )

    plt.title(
        "Condition Numbers"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, f"condition_numbers_{RUN_TIMESTAMP}.png"),
        dpi=300,
    )

    plt.close()

    # condition numbers last 200,000 steps, linear

    condition_steps = np.array(
        condition_steps
    )

    cond_mean_history = np.array(
        cond_mean_history
    )

    cond_max_history = np.array(
        cond_max_history
    )

    if len(condition_steps) > 0:

        last_step = condition_steps[-1]

        start_step = max(
            0,
            last_step - 200_000,
        )

        mask = (
            condition_steps
            >= start_step
        )

        plt.figure(figsize=(8, 5))

        plt.plot(
            condition_steps[mask],
            cond_mean_history[mask],
            label="Mean Cond",
        )

        plt.plot(
            condition_steps[mask],
            cond_max_history[mask],
            label="Max Cond",
        )

        # no plt.yscale("log") here, linear y

        plt.legend()

        plt.xlabel("Gossip Step")

        plt.ylabel(
            "Condition Number"
        )

        plt.title(
            "Condition Numbers — Last 200,000 Steps"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(outdir, f"condition_numbers_last_200k_linear_{RUN_TIMESTAMP}.png"),
            dpi=300,
        )

        plt.close()

    plt.figure(figsize=(8, 5))

    plt.plot(
        condition_steps,
        stable_fraction_history,
    )

    plt.xlabel("Gossip Step")

    plt.ylabel(
        "Stable Fraction"
    )

    plt.title(
        "Stable Agents Fraction"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, f"stable_fraction_{RUN_TIMESTAMP}.png"),
        dpi=300,
    )

    plt.close()

def plot_matrix_history(
    matrix_history,
    steps_history,
    experiment_name,
    base_dir="results",
):
    outdir = os.path.join(base_dir, experiment_name)
    ensure_dir(outdir)

    matrix_history = np.array(
        matrix_history
    )

    plt.figure(figsize=(8, 6))

    for i in range(
        matrix_history.shape[1]
    ):

        for j in range(
            matrix_history.shape[2]
        ):

            plt.plot(
                steps_history,
                matrix_history[:, i, j],
            )

    plt.xlabel("Gossip Step")

    plt.ylabel(
        "Matrix Entry"
    )

    plt.title(
        "Average Stable Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, f"matrix_history_{RUN_TIMESTAMP}.png"),
        dpi=300,
    )

    plt.close()

def plot_theta_history(
    theta_history,
    steps_history,
    theta_target,
    experiment_name,
    base_dir="results",
):
    outdir = os.path.join(base_dir, experiment_name)
    ensure_dir(outdir)

    theta_history = np.array(
        theta_history
    )

    p = theta_history.shape[1]

    plt.figure(
        figsize=(10, 6)
    )

    for k in range(p):

        plt.plot(
            steps_history,
            theta_history[:, k],
            label=f"Theta {k}",
        )

        plt.axhline(
            theta_target[k],
            linestyle="--",
        )

    plt.xlabel("Gossip Step")

    plt.ylabel("Value")

    plt.title(
        "Theta Convergence"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, f"theta_history_{RUN_TIMESTAMP}.png"),
        dpi=300,
    )

    plt.close()

def plot_regression_metrics(metrics, experiment_name, base_dir="results"):
    outdir = os.path.join(base_dir, experiment_name)
    ensure_dir(outdir)

    splits = [s for s in ["Seen", "Unseen"] if s in metrics]
    if not splits:
        return  # nothing to plot

    x = np.arange(len(splits))

    # RMSE
    rmse = [metrics[s]["RMSE"] for s in splits]
    plt.figure(figsize=(8, 5))
    plt.bar(x, rmse)
    plt.xticks(x, splits)
    plt.ylabel("RMSE")
    plt.title("RMSE")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"rmse_{RUN_TIMESTAMP}.png"), dpi=300)
    plt.close()

    # MAE
    mae = [metrics[s]["MAE"] for s in splits]
    plt.figure(figsize=(8, 5))
    plt.bar(x, mae)
    plt.xticks(x, splits)
    plt.ylabel("MAE")
    plt.title("MAE")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"mae_{RUN_TIMESTAMP}.png"), dpi=300)
    plt.close()

    # R2
    r2 = [metrics[s]["R2"] for s in splits]
    plt.figure(figsize=(8, 5))
    plt.bar(x, r2)
    plt.xticks(x, splits)
    plt.ylabel("R²")
    plt.title("R²")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"r2_{RUN_TIMESTAMP}.png"), dpi=300)
    plt.close()

def plot_norm_history(
    l1_history,
    l2_history,
    linf_history,
    norm_steps,
    experiment_name,
    base_dir="results",
):
    outdir = os.path.join(base_dir, experiment_name)
    ensure_dir(outdir)

    plt.figure(figsize=(8, 5))

    plt.plot(
        norm_steps,
        l1_history,
        label="L1",
    )

    plt.plot(
        norm_steps,
        l2_history,
        label="L2",
    )

    plt.plot(
        norm_steps,
        linf_history,
        label="L∞",
    )

    plt.xlabel("Gossip Step")

    plt.ylabel(
        "Average Norm of Change in Theta"
    )

    plt.title(
        "Convergence of Local Theta Vectors"
    )

    plt.legend()

    plt.yscale("log")

    plt.tight_layout()

    plt.savefig(
        os.path.join(outdir, f"theta_norm_convergence_{RUN_TIMESTAMP}.png"),
        dpi=300,
    )

    plt.close()
