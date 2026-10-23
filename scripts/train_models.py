"""Train and compare the project's regression and classification models."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from student_performance.training import train_experiments

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/raw/student-mat.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts")
    args = parser.parse_args()
    result = train_experiments(args.data, args.output)
    print(json.dumps({"rows": result["dataset_rows"], "regression_best": result["regression"]["best_model"], "regression_metrics": result["regression"]["best_metrics"], "classification_best": result["classification"]["best_model"], "classification_metrics": result["classification"]["best_metrics"]}, indent=2))
