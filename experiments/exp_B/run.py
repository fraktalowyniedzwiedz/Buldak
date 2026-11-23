from pathlib import Path
import itertools
import subprocess
import sys
import shutil
import json
from datetime import datetime

EXP_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = EXP_DIR.parents[1]
MAIN_SCRIPT = PROJECT_ROOT / "main.py"
if not MAIN_SCRIPT.is_file():
    raise FileNotFoundError(f"Master-directory main.py not found: {MAIN_SCRIPT}")
DATA_DIR = EXP_DIR / "data"
RESULTS_DIR = EXP_DIR / "results" / f"exp_B_{datetime.now():%Y%m%d_%H%M%S}"

A_VALUES = [100, 500, 1000]
D_VALUES = [2, 5, 10]
SEEDS = range(1, 20)

subprocess.run([sys.executable, str(EXP_DIR / "generate_data.py")], check=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
print(f"Results will be saved under: {RESULTS_DIR}")
(RESULTS_DIR / "data").mkdir()
(RESULTS_DIR / "experiment.json").write_text(json.dumps({
    "gossip_mode": "full", "steps": 200000,
    "seeds": list(SEEDS), "A_values": A_VALUES, "D_values": D_VALUES,
    "early_stopping_requested": True,
}, indent=2))

for A, D, seed in itertools.product(A_VALUES, D_VALUES, SEEDS):
    exp_name = f"A{A}_D{D}_seed{seed}"
    data_file = DATA_DIR / f"{exp_name}.csv"
    archived = RESULTS_DIR / "data" / data_file.name
    shutil.copy2(data_file, archived)
    cmd = [
        sys.executable, str(MAIN_SCRIPT),
        "--data_file", str(archived),
        "--seen_fraction", "1.0",
        "--seed", str(seed),
        "--steps", "200000",
        "--gossip_mode", "full",
        "--stop_when_done",
        "--epsilon", "1e-10",
        "--r", "20",
        "--cond_threshold", "50",
        "--output_dir", str(RESULTS_DIR),
        "--experiment_name", exp_name,
    ]
    print(f"Running {exp_name}")
    subprocess.run(cmd, check=True)

print("All runs completed.")
