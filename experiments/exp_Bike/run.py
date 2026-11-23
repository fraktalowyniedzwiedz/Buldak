from __future__ import annotations

import argparse
from datetime import datetime
from io import BytesIO
import json
from pathlib import Path
import shutil
import sys
import time
import warnings
from urllib.request import Request, urlopen
import zipfile

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
from simulation import DistributedRegression

UCI_URLS = (
    "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip",
    "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip",
)
NUMERIC = ("yr", "hr", "holiday", "workingday", "temp", "atemp", "hum", "windspeed")
CATEGORICAL = ("season", "weathersit", "weekday")


def fraction_tag(f):
    return f"{int(round(f * 100)):03d}"


def csv_path(granularity, fraction, seed):
    directory = HERE / "data" / granularity / "full_design"
    return directory / f"BS_Seen{fraction_tag(fraction)}_seed{seed}.csv"


def obtain_raw_csv(granularity, raw_csv=None):
    if raw_csv is not None:
        source = Path(raw_csv).expanduser().resolve()
        if not source.is_file():
            raise FileNotFoundError(source)
        return source
    target = HERE / "data" / "raw" / f"{granularity}.csv"
    if target.is_file():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in UCI_URLS:
        try:
            print(f"Downloading Bike Sharing raw data: {url}", flush=True)
            request = Request(url, headers={"User-Agent": "Python research experiment"})
            with urlopen(request, timeout=90) as response:
                archive = BytesIO(response.read())
            with zipfile.ZipFile(archive) as z:
                matches = [name for name in z.namelist()
                           if Path(name).name.lower() == f"{granularity}.csv"]
                if not matches:
                    raise FileNotFoundError(f"{granularity}.csv missing from archive")
                tmp = target.with_suffix(".csv.tmp")
                with z.open(matches[0]) as src, tmp.open("wb") as dest:
                    shutil.copyfileobj(src, dest)
                tmp.replace(target)
            return target
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    raise RuntimeError("Could not download the UCI Bike Sharing dataset. "
                       "Provide --raw_csv /path/to/hour.csv (or day.csv).\n"
                       + "\n".join(errors))


def load_raw_features(raw_csv, granularity):
    raw = pd.read_csv(raw_csv)
    if "cnt" not in raw.columns:
        raise ValueError(f"{raw_csv}: missing target column 'cnt'")
    numeric = [c for c in NUMERIC if c in raw.columns]
    expected = [c for c in NUMERIC if granularity == "hour" or c != "hr"]
    missing = [c for c in expected + list(CATEGORICAL) if c not in raw.columns]
    if missing:
        raise ValueError(f"{raw_csv}: missing required features: {missing}")
    numeric = [c for c in expected if c in numeric]
    X_num = raw[numeric].to_numpy(dtype=float)
    dummy = pd.get_dummies(raw[list(CATEGORICAL)].astype(int).astype(str),
                           prefix=list(CATEGORICAL), drop_first=False)
    X_cat = dummy.to_numpy(dtype=float)
    X = np.column_stack((X_num, X_cat))
    y = raw["cnt"].to_numpy(dtype=float)
    if len(y) < 3 or not np.isfinite(X).all() or not np.isfinite(y).all():
        raise ValueError("Dataset needs >=3 finite observations")
    return X, y, numeric + list(dummy.columns)


def prepare_datasets(args):
    required = {(fraction, seed): csv_path(args.granularity, fraction, seed)
                for fraction in args.seen_fractions for seed in args.seeds}
    meta_path = next(iter(required.values())).parent / "features.json"
    expected_grid = {"seen_fractions": sorted(args.seen_fractions),
                     "seeds": sorted(args.seeds), "encoding": "full_design", "audit_version": 1}
    matches_manifest = False
    if meta_path.exists():
        try:
            previous = json.loads(meta_path.read_text())
            matches_manifest = all(previous.get(key) == value
                                   for key, value in expected_grid.items())
            if args.raw_csv is not None:
                matches_manifest = (matches_manifest and
                                    previous.get("raw_source") ==
                                    str(args.raw_csv.expanduser().resolve()))
        except (OSError, ValueError):
            pass
    if not args.regenerate and matches_manifest and all(path.is_file() for path in required.values()):
        print(f"Reusing {len(required)} pre-generated CSV datasets.", flush=True)
        return required

    raw_path = obtain_raw_csv(args.granularity, args.raw_csv)
    X, y, feature_names = load_raw_features(raw_path, args.granularity)
    masks = {}
    for fraction, seed in required:
        idx = np.arange(len(y))
        if fraction == 1.0:
            seen_idx = idx
        else:
            seen_idx, _ = train_test_split(idx, train_size=fraction,
                                           random_state=seed)
        mask = np.zeros(len(y), dtype=bool)
        mask[seen_idx] = True
        masks[fraction, seed] = mask

    print(f"Original {args.granularity} design: n={len(y)}, "
          f"features={X.shape[1]}, augmented dimension={X.shape[1]+1} "
          "(no feature selection or rank checks)", flush=True)

    scalers = {}
    for (fraction, seed), path in required.items():
        mask = masks[fraction, seed]
        scaler = StandardScaler().fit(X[mask])
        X_scaled = scaler.transform(X)
        scalers[path.name] = {"mean": scaler.mean_.tolist(), "scale": scaler.scale_.tolist()}
        frame = pd.DataFrame(X_scaled, columns=[f"x{i}" for i in range(X.shape[1])])
        frame["y"] = y
        frame["split"] = np.where(mask, "seen", "unseen")
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"  dataset: {path.name} | seen={mask.sum()} unseen={(~mask).sum()}",
              flush=True)
    manifest = {**expected_grid,
                "raw_source": str(raw_path), "scalers": scalers,
                "feature_names": feature_names,
                "original_candidate_names": list(feature_names),
                "excluded_features": [],
                "selection": "none; full original design retained regardless of rank",
                "intercept_appended_in_simulation": True}
    meta_path.write_text(json.dumps(manifest, indent=2))
    return required


def read_prepared(path):
    df = pd.read_csv(path)
    columns = sorted((col for col in df.columns if col.startswith("x") and
                      col[1:].isdigit()), key=lambda x: int(x[1:]))
    if not columns or columns != [f"x{i}" for i in range(len(columns))]:
        raise ValueError(f"{path}: missing or noncontiguous x0, x1, ... columns")
    if not {"y", "split"}.issubset(df.columns):
        raise ValueError(f"{path}: expected y and split columns")
    split = df["split"].to_numpy()
    if not np.isin(split, ["seen", "unseen"]).all():
        raise ValueError(f"{path}: only 'seen' and 'unseen' split values allowed")
    seen = split == "seen"
    if not seen.any():
        raise ValueError(f"{path}: zero seen observations")
    X = df[columns].to_numpy(dtype=float)
    y = df["y"].to_numpy(dtype=float)
    Z = np.column_stack((X, np.ones(len(X))))
    if not np.isfinite(Z).all() or not np.isfinite(y).all():
        raise ValueError(f"{path}: nonfinite observations")
    return Z, y, seen


def normal_json_number(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def prediction_metrics(Z, y, theta):
    if not len(y):
        return None
    predicted = Z @ theta
    return {"RMSE": float(np.sqrt(mean_squared_error(y, predicted))),
            "MAE": float(mean_absolute_error(y, predicted)),
            "R2": float(r2_score(y, predicted)) if len(y) > 1 else None}


def objective_setup(Z, y):
    theta_center = np.linalg.lstsq(Z, y, rcond=None)[0]

    Q, R = np.linalg.qr(Z, mode="reduced")
    projected_y = Q.T @ y
    orthogonal_residual = y - Q @ projected_y
    orthogonal_sq = float(orthogonal_residual @ orthogonal_residual)
    centralized_residual = Z @ theta_center - y
    minimum = float(centralized_residual @ centralized_residual)
    return minimum, (R, projected_y, orthogonal_sq)


def objective_values(theta_agents, usable, minimum, factor):
    finite_eligible = np.asarray(usable, dtype=bool) & np.isfinite(theta_agents).all(axis=1)
    sse = np.full(len(theta_agents), np.nan)
    excess = np.full(len(theta_agents), np.nan)
    if finite_eligible.any():
        R, projected_y, orthogonal_sq = factor
        projected_residual = theta_agents[finite_eligible] @ R.T - projected_y
        values = np.einsum('ni,ni->n', projected_residual, projected_residual,
                           optimize=True) + orthogonal_sq
        sse[finite_eligible] = values
        raw_excess = values - minimum
        tolerance = 1e-8 * max(minimum, 1.0)
        if np.any(raw_excess < -tolerance):
            warnings.warn(
                "Numerical issue: an agent's computed SSE lies below the "
                "centralized least-squares minimum. Check design rank and "
                "conditioning. Raw SSE is retained, negative excess "
                "retained in signed gap summaries.",
                RuntimeWarning, stacklevel=2,
            )
        excess[finite_eligible] = raw_excess
    return sse, excess, finite_eligible


def checkpoint(sim, step, objectives, archive):
    theta, conditions, stable = sim.compute_theta_agents()
    M_diff = sim.agents_M - sim.M_global
    B_diff = sim.agents_B - sim.B_global
    p = sim.p
    initial_M_sum = sim.w_initial_sum[:p*p].reshape(p, p)
    initial_B_sum = sim.w_initial_sum[p*p:]
    mass_drift = np.sqrt(np.square(sim.agents_M.sum(axis=0) - initial_M_sum).sum()
                         + np.square(sim.agents_B.sum(axis=0) - initial_B_sum).sum())
    row = {
        "step": step,
        "theta_error": float(sim.relative_theta_error(theta)),
        "E_M": float(np.mean(np.sum(np.square(M_diff), axis=(1, 2))) /
                     (np.square(sim.M_global).sum() + 1e-12)),
        "E_B": float(np.mean(np.sum(np.square(B_diff), axis=1)) /
                     (np.square(sim.B_global).sum() + 1e-12)),
        "mass_drift": float(mass_drift),
        "cond_mean": float(np.mean(conditions)),
        "cond_max": float(np.max(conditions)),
        "stable_fraction": float(np.mean(stable)),
    }
    finite_any = np.asarray(stable, bool) & np.isfinite(theta).all(axis=1)
    raw = {"agent": np.flatnonzero(finite_any), "n_agents": sim.N}
    row["n_agents"] = sim.N
    row["objective_usable_count"] = int(finite_any.sum())
    row["objective_usable_fraction"] = float(np.mean(finite_any))
    for tag, (minimum, factor) in objectives.items():
        sse, excess, _ = objective_values(theta, stable, minimum, factor)
        if not np.isfinite(sse[finite_any]).all() or not np.isfinite(excess[finite_any]).all():
            raise FloatingPointError(f"Nonfinite objective at step {step} on {tag}")
        raw[f"sse_{tag}"] = sse[finite_any]
        raw[f"signed_gap_{tag}"] = excess[finite_any]
        raw[f"central_min_{tag}"] = minimum
        gaps = excess[finite_any]
        row[f"negative_gaps_{tag}"] = int((gaps < 0).sum())
        row[f"zero_gaps_{tag}"] = int((gaps == 0).sum())
        for quantile in (25, 75):
            row[f"gap_q{quantile}_{tag}"] = np.percentile(gaps, quantile) if gaps.size else np.nan
        good_sse = sse[np.isfinite(sse)]
        good_excess = excess[np.isfinite(excess)]
        row[f"central_min_{tag}"] = minimum
        row[f"agent_sse_mean_{tag}"] = float(np.mean(good_sse)) if len(good_sse) else np.nan
        row[f"agent_sse_median_{tag}"] = float(np.median(good_sse)) if len(good_sse) else np.nan
        row[f"agent_sse_min_{tag}"] = float(np.min(good_sse)) if len(good_sse) else np.nan
        row[f"agent_sse_max_{tag}"] = float(np.max(good_sse)) if len(good_sse) else np.nan
        row[f"objective_gap_mean_{tag}"] = (float(np.mean(good_excess))
                                              if len(good_excess) else np.nan)
        row[f"objective_gap_median_{tag}"] = (float(np.median(good_excess))
                                                if len(good_excess) else np.nan)
        row[f"objective_ratio_mean_{tag}"] = (float(np.mean(good_sse) / minimum)
                                                if len(good_sse) and minimum > 0 else np.nan)
    np.savez_compressed(archive / f"step_{step}.npz", **raw)
    return row, theta, conditions, stable


def simulate(csv, output, seed, mode, args):
    np.random.seed(seed)
    Z, y, seen = read_prepared(csv)
    sim = DistributedRegression(
        X_seen_aug=Z[seen], y_seen=y[seen],
        X_global_aug=Z, y_global=y,
        cond_threshold=args.cond_threshold, gossip_mode=mode,
        stop_on_theta_change=False, epsilon=1e-10, r=20,
    )
    objectives = {}
    for tag, select in (("seen", seen), ("all", np.ones(len(y), dtype=bool))):
        objectives[tag] = objective_setup(Z[select], y[select])
    archive = output / "agent_objectives"
    archive.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    convergence = []
    final = None
    for step in range(1, args.steps + 1):
        sim.gossip_step()
        log_now = (step == 1 or step % args.checkpoint_every == 0
                   or step % args.progress_every == 0 or step == args.steps)
        if not log_now:
            continue
        point, theta, conditions, stable = checkpoint(sim, step, objectives, archive)
        convergence.append(point)
        if step == 1 or step % args.progress_every == 0 or step == args.steps:
            print(f"  step {step:,}/{args.steps:,} | E_theta={point['theta_error']:.3e} "
                  f"| usable={point['stable_fraction']:.1%} "
                  f"| SSE_seen(mean/min)={point['agent_sse_mean_seen']:.4g}"
                  f"/{point['central_min_seen']:.4g} "
                  f"| elapsed={time.monotonic()-start:.1f}s", flush=True)
        if step == args.steps:
            final = (theta, conditions, stable, point)
    pd.DataFrame(convergence).to_csv(output / "convergence.csv", index=False)
    theta, conditions, stable, point = final
    valid = np.asarray(stable, dtype=bool) & np.isfinite(theta).all(axis=1)
    per_agent = pd.DataFrame({
        "agent": np.arange(sim.N), "stable": stable, "finite_solution": np.isfinite(theta).all(axis=1),
        "condition_number": conditions,
    })
    for tag, (minimum, factor) in objectives.items():
        sse, excess, _ = objective_values(theta, stable, minimum, factor)
        per_agent[f"sse_{tag}"] = sse
        per_agent[f"central_min_{tag}"] = minimum
        per_agent[f"excess_{tag}"] = excess
        per_agent[f"ratio_{tag}"] = sse / minimum if minimum > 0 else np.nan
    per_agent.to_csv(output / "final_agent_objectives.csv", index=False)
    theta_mean = theta[valid].mean(axis=0) if valid.any() else None
    metrics = {
        "seed": seed, "steps": args.steps, "mode": mode,
        "n_agents": int(sim.N), "p_aug": int(sim.p),
        "n_stable": int(np.sum(stable)), "n_finite_eligible": int(np.sum(valid)),
        "final_stable_fraction": float(np.mean(stable)),
        "objective_usable_fraction": float(np.mean(valid)),
        "rel_error_seen": None, "rel_error_global": None,
        "Seen": None, "Unseen": None,
        "central_min_seen": objectives["seen"][0],
        "central_min_all": objectives["all"][0],
    }
    for field in ("agent_sse_mean_seen", "objective_gap_mean_seen", "objective_ratio_mean_seen",
                  "agent_sse_mean_all", "objective_gap_mean_all", "objective_ratio_mean_all"):
        metrics[field] = normal_json_number(point[field])
    if theta_mean is not None:
        for name, reference in (("rel_error_seen", sim.theta_seen),
                                ("rel_error_global", sim.theta_global)):
            norm = np.linalg.norm(reference)
            metrics[name] = normal_json_number(np.linalg.norm(theta_mean - reference) / norm) if norm > 0 else None
        metrics["Seen"] = prediction_metrics(Z[seen], y[seen], theta_mean)
        metrics["Unseen"] = prediction_metrics(Z[~seen], y[~seen], theta_mean)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False))
    print(f"  saved {output} | final usable={float(stable.mean()):.1%}, "
          f"finite eligible={valid.sum()}/{sim.N}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--granularity", choices=["hour", "day"], default="hour")
    parser.add_argument("--raw_csv", type=Path, help="Existing UCI hour.csv or day.csv; otherwise download/cache")
    parser.add_argument("--seen_fractions", type=float, nargs="+", default=[0.10, 0.25, 0.50, 1.0])
    parser.add_argument("--modes", choices=["full", "coordinate"], nargs="+", default=["full"])
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(1, 21)))
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--cond_threshold", type=float, default=1e18)
    parser.add_argument("--checkpoint_every", type=int, default=5000)
    parser.add_argument("--progress_every", type=int, default=5000)
    parser.add_argument("--generate_only", action="store_true")
    parser.add_argument("--regenerate", action="store_true")
    args = parser.parse_args()
    if not args.seen_fractions or not args.seeds or not args.modes:
        parser.error("at least one fraction, seed and mode required")
    if any(not 0 < f <= 1 for f in args.seen_fractions):
        parser.error("seen fractions must be in (0, 1]")
    if len({fraction_tag(f) for f in args.seen_fractions}) != len(args.seen_fractions):
        parser.error("distinct seen fractions must produce distinct percentage tags")
    if min(args.steps, args.checkpoint_every, args.progress_every) < 1:
        parser.error("step count and reporting intervals must be positive")
    if args.cond_threshold < 0:
        parser.error("condition threshold must be nonnegative")

    paths = prepare_datasets(args)
    if args.generate_only:
        print("Dataset generation complete.")
        return
    output = HERE / "results" / ("exp_BS_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    output.mkdir(parents=True, exist_ok=True)
    snapshot = output / "designs"
    snapshot.mkdir()
    for key, csv in paths.items():
        paths[key] = Path(shutil.copy2(csv, snapshot))
    shutil.copy2(csv.parent / "features.json", snapshot)
    first = next(iter(paths.values()))
    Z, y, _ = read_prepared(first)
    config = {
        "experiment": "exp_Bike", "granularity": args.granularity,
        "feature_encoding": "full_design",
        "features_manifest": str((next(iter(paths.values())).parent / "features.json").resolve()),
        "seen_fractions": args.seen_fractions, "gossip_modes": args.modes,
        "seeds": args.seeds, "steps": args.steps, "p_aug": int(Z.shape[1]),
        "cond_threshold": args.cond_threshold, "checkpoint_every": args.checkpoint_every,
        "progress_every": args.progress_every, "stop_on_theta_change": False,
        "data_dir": str(next(iter(paths.values())).parent.resolve()),
        "objective_reference": "centralized np.linalg.lstsq on matching seen/all design matrix",
        "communication_cost_full": 2 * (Z.shape[1] * (Z.shape[1] + 1) // 2 + Z.shape[1]),
        "communication_cost_coordinate": 2,
    }
    (output / "_config.json").write_text(json.dumps(config, indent=2))
    total = len(args.seen_fractions) * len(args.modes) * len(args.seeds)
    i = 0
    for fraction in args.seen_fractions:
        for mode in args.modes:
            for seed in args.seeds:
                i += 1
                name = f"BS_Seen{fraction_tag(fraction)}_Mode{'full' if mode == 'full' else 'coord'}_seed{seed}"
                print(f"\n[{i}/{total}] {name} | {args.steps:,} fixed interactions", flush=True)
                simulate(paths[fraction, seed], output / name, seed, mode, args)
    print(f"\nCompleted: {output}", flush=True)


if __name__ == "__main__":
    main()
