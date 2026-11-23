import argparse
from pathlib import Path
import sys
import pandas as pd

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR.parent))
from exp_E.prescribed import PrescribedConditionDataset


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--A', type=int, default=100)
    p.add_argument('--D', type=int, default=5)
    p.add_argument('--noise', type=float, default=0.1)
    p.add_argument('--seeds', type=int, nargs='+', default=list(range(1, 11)))
    p.add_argument('--kappas', type=float, nargs='+',
                   default=[1.0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6])
    args = p.parse_args()
    outdir = EXP_DIR / 'data'
    outdir.mkdir(parents=True, exist_ok=True)
    count = 0
    for kappa in args.kappas:
        for seed in args.seeds:
            ds = PrescribedConditionDataset(
                n_samples=args.A, n_features=args.D,
                kappa_target=kappa, noise=args.noise,
                random_state=seed, seen_fraction=1.0, standardize=True,
            )
            frame = pd.DataFrame(ds.X_seen, columns=[f'x{i}' for i in range(args.D)])
            frame['y'] = ds.y_seen
            path = outdir / f'A{args.A}_D{args.D}_kappa{kappa:g}_seed{seed}.csv'
            frame.to_csv(path, index=False)
            count += 1
    print(f'Generated {count} prescribed CSV datasets in {outdir}')


if __name__ == '__main__':
    main()
