"""Run the complete Retail Demand Forecasting workflow from raw data to reports.

Run from the project root:
    python run_pipeline.py

The LSTM and rolling-origin stages are intentionally included because both are
part of the project's evaluation protocol. They can be lengthy on CPU.
"""
from pathlib import Path
import argparse
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run_script(script: Path):
    print("\n" + "=" * 80, flush=True)
    print(f"RUNNING: {script.relative_to(ROOT)}", flush=True)
    print("=" * 80, flush=True)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description="Run retail forecasting end-to-end")
    parser.add_argument("--skip-rolling-validation", action="store_true", help="Skip the 3-fold rolling-origin backtest")
    args = parser.parse_args()

    required = [ROOT / "data/raw/train.csv", ROOT / "data/raw/store.csv"]
    missing = [path for path in required if not path.exists()]
    if missing:
        formatted = "\n".join(f"  - {path.relative_to(ROOT)}" for path in missing)
        raise SystemExit(
            "Required Rossmann files are missing. Download train.csv and store.csv from Kaggle, "
            f"then place them under data/raw/.\nMissing files:\n{formatted}"
        )

    stages = [
        ROOT / "src/01_data_ingestion_eda.py",
        ROOT / "src/02_feature_engineering.py",
        ROOT / "src/03_classical_modeling.py",
        ROOT / "src/04_ml_modeling.py",
    ]
    stages.append(ROOT / "src/05_dl_modeling.py")
    if not args.skip_rolling_validation:
        stages.append(ROOT / "src/07_rolling_origin_validation.py")
    stages.append(ROOT / "src/06_evaluation.py")

    for script in stages:
        run_script(script)

    print("\nPipeline complete. Start the dashboard with:")
    print("  streamlit run dashboard/app.py")
    print(f"Reports and artifacts: {ROOT / 'data/processed'} and {ROOT / 'models'}")


if __name__ == "__main__":
    main()
