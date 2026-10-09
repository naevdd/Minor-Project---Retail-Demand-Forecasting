"""Run the updated forecasting benchmark end-to-end.

Run from the project root:
    python run_pipeline.py

The modeling stages can be long, especially LSTM. Each stage writes its own
results under data/features and the final benchmark under data/processed.
"""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = [
    ROOT / "src" / "03_classical_modeling.py",
    ROOT / "src" / "04_ml_modeling.py",
    ROOT / "src" / "05_dl_modeling.py",
    ROOT / "src" / "06_evaluation.py",
]

for script in SCRIPTS:
    print("\n" + "=" * 80)
    print(f"RUNNING: {script.relative_to(ROOT)}")
    print("=" * 80)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, check=True)

print("\nPipeline complete. Launch the dashboard with:")
print("  streamlit run dashboard/app.py")
