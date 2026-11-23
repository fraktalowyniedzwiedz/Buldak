from pathlib import Path
import itertools

import pandas as pd
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

EXP_DIR = Path(__file__).resolve().parent
DATA_DIR = EXP_DIR / "data"

A_VALUES = [100, 500, 1000]
D_VALUES = [2, 5, 10]
SEEDS = range(1, 20)
NOISE = 0.1
EFFECTIVE_RANK = None


def generate_one(n_samples: int, n_features: int, seed: int) -> pd.DataFrame:
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=NOISE,
        random_state=seed,
        effective_rank=EFFECTIVE_RANK,
        coef=False,
    )

    X = StandardScaler().fit_transform(X)

    frame = pd.DataFrame(X, columns=[f"x{i}" for i in range(n_features)])
    frame["y"] = y
    return frame


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    for A, D, seed in itertools.product(A_VALUES, D_VALUES, SEEDS):
        path = DATA_DIR / f"A{A}_D{D}_seed{seed}.csv"
        generate_one(A, D, seed).to_csv(path, index=False)
        count += 1
    print(f"Generated {count} CSV datasets in {DATA_DIR}")


if __name__ == "__main__":
    main()
