"""Download the UCI Student Performance mathematics dataset."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from student_performance.data import download_dataset

if __name__ == "__main__":
    path = download_dataset()
    print(f"Dataset downloaded to {path}")
