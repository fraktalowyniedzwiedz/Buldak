# Gossip-Based Linear Regression Inspired by Population Protocols

Experimental code accompanying a master’s thesis at the University of Wrocław.

## Getting started

Run these commands from the repository root in a POSIX shell.

```sh
python3.12 -m venv .venv
. .venv/bin/activate
python -m pip install numpy==2.3.5 pandas==2.2.3 matplotlib==3.10.8 scikit-learn==1.8.0
```

## Reproducing Chapter 3

Use an archived run to regenerate a thesis figure or table, or run an experiment from its input data. The [figure and table index](#figure-and-table-index) lists the output files for each thesis item.

`run.py` prepares the data, runs the simulation, and writes its logs. `analyze.py` reads a run directory and writes the aggregate plots and tables there. It can replace previously generated summaries in that directory.

### Start from archived results

Place each archived experiment folder under `experiments/`, preserving its `results/` tree. Run the line for the result of interest. The paths select the archived inputs even if a more recent run exists.

```sh
python experiments/exp_B/analyze.py experiments/exp_B/results/exp_B_20261120_205519
python experiments/exp_E/analyze.py experiments/exp_E/results/exp_E_20261120_205717
python experiments/exp_F/analyze.py experiments/exp_F/results/exp_F_20261109_203534
python experiments/exp_G/analyze.py experiments/exp_G/results/exp_G_20261120_205842_845873
python experiments/exp_H/analyze.py experiments/exp_H/results/exp_H_20261118_162431
```

### Start from the data

For a full run, choose one experiment and run both lines below. The example uses B, set the shell variable to `exp_E`, `exp_F`, `exp_G`, `exp_H`, or `exp_Bike` for the other experiments. The analyzer defaults to the latest eligible run in that experiment’s results directory. To select a specific new run, pass the path printed by its runner.

```sh
experiment=exp_B
python "experiments/$experiment/run.py"
python "experiments/$experiment/analyze.py"
```

| Experiment | Configuration used by the runner defaults |
|---|---|
| B · correctness and scaling | Full-state gossip. All nine combinations of 100, 500, or 1,000 agents and 2, 5, or 10 features. Seeds 1–19, up to 200,000 interactions per run. Stop when all agents are inactive or the budget is reached. |
| E · conditioning | 100 agents, 5 features, 7 target condition numbers from 1 to 1,000,000 in powers of ten. Coordinate gossip, seeds 1–10, up to 200,000 interactions with an exact terminal checkpoint, solve cutoff 10¹². The direct-perturbation control separately uses seeds 1–20 and four equal matrix/vector perturbation levels from 10⁻⁶ to 10⁻³. |
| F · observation noise | 100 observations, 2 features, 80 training agents. Ten noise levels (0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10), seeds 1–20, coordinate gossip, 200,000 interactions. |
| G · heterogeneous clients | 100 agents with 10 observations each, 2 features, two groups of 50, noise 0.1. Coordinate gossip, seeds 1–20, stop when all agents are inactive or at 200,000 interactions. |
| H · California Housing | Four nested seen fractions (0.10, 0.25, 0.50, 1.00), full-state and scalar-coordinate gossip, seeds 1–3, 1,000,000 interactions. First-time data preparation uses scikit-learn’s California Housing download. |
| Bike · Bike Sharing | Hourly UCI Bike Sharing data. Four seen fractions (0.10, 0.25, 0.50, 1.00), full-state gossip, seeds 1–20, 1,000,000 interactions per run. Condition-number cutoff of 10¹⁸ for local solves. |

### Figure and table index

Paths in this table are relative to the selected run directory, the filenames are the outputs of the analyzers.

| Thesis item | Experiment | Output and how to read it |
|---|---|---|
| Fig. 3.1 | B | `analysis_figures/convergence_A1000_D10.png`. Four diagnostics averaged over the seeds recorded at each checkpoint. |
| Table 3.1 | B | `analysis_figures/final_stability_runs.csv`. Group the final logged parameter errors by agent count and dimension. |
| Fig. 3.2 | B | `analysis_figures/Phi_norm_collapse.png`. The plotted quantity is Ψ. Each trajectory is divided by its own initial disagreement before averaging, with interaction count scaled by A−1. |
| Fig. 3.3 | E | `analysis/gossip/Etheta_vs_sqrt_EM2.png`. The points use the target-crossing records in `analysis/gossip/amplification_runs.csv` and their aggregates. The separate direct control writes `perturbation/perturbation_runs.csv` and `analysis/perturbation_aggregate.csv`. |
| Figs. 3.4 and 3.5 | F | `convergence_trajectories.png` and `final_theta_vs_test_rmse.png`, respectively. Endpoint means and standard deviations are in `noise_summary.csv`. |
| Fig. 3.6 | G | `heterogeneity_comparison.png` and `error_comparison.csv`. The default thesis profile plots the mean individual-agent error at the stopping interaction. |
| Fig. 3.7 | H | `analysis_figures/ca_communication_tradeoff.png`. The horizontal axis counts 108 numerical scalars per full-state interaction. |
| Table 3.2 | H | `analysis_figures/ca_prediction_table.tex` and `analysis_figures/ca_paired_predictions.csv`. The underlying seed-level scores are in `analysis_figures/ca_predictions_by_seed.csv`. |
| Fig. 3.8 | Bike | `analysis_figures/bs_objective_comparison_seen.png`, with one `bs_plotted_seen_*_full.csv` per fraction. The archived `bs_objective_comparison_seenv2.png` matches the published figure pixel for pixel. |

## Repository layout

The repository root contains `simulation.py` (gossip updates and local solves), `main.py` (shared CSV-driven runner), and `plotting.py` (single-run plots). Under `experiments/`, each experiment has its own `run.py` and `analyze.py`, with some having helpers.
