"""Dataset loading, validation, and target preparation."""

from pathlib import Path
from io import BytesIO
from urllib.request import urlretrieve

import pandas as pd

DATA_URL = "https://archive.ics.uci.edu/static/public/320/student+performance.zip"
DEFAULT_DATA_PATH = Path("data/raw/student-mat.csv")
TARGET = "G3"
LEAKAGE_COLUMNS = {"G1", "G2", "G3"}


def download_dataset(destination: Path = DEFAULT_DATA_PATH) -> Path:
    """Download and extract the UCI zip archive's mathematics CSV."""
    from zipfile import ZipFile

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    archive = destination.parent / "student-performance.zip"
    urlretrieve(DATA_URL, archive)
    with ZipFile(archive) as zipped:
        member = next((n for n in zipped.namelist() if n.endswith("student-mat.csv")), None)
        if member is not None:
            destination.write_bytes(zipped.read(member))
        else:
            # UCI currently wraps the CSV files inside a nested student.zip.
            nested_name = next((n for n in zipped.namelist() if n.lower().endswith("student.zip")), None)
            if nested_name is None:
                raise FileNotFoundError("student-mat.csv or nested student.zip was not found in the UCI archive")
            with ZipFile(BytesIO(zipped.read(nested_name))) as nested:
                member = next((n for n in nested.namelist() if n.endswith("student-mat.csv")), None)
                if member is None:
                    raise FileNotFoundError("student-mat.csv was not found inside nested student.zip")
                destination.write_bytes(nested.read(member))
    archive.unlink(missing_ok=True)
    return destination


def load_data(path: Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and validate the semicolon-separated UCI mathematics dataset."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run: python scripts/download_data.py")
    frame = pd.read_csv(path, sep=";")
    required = LEAKAGE_COLUMNS
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Dataset contains no rows")
    return frame


def prepare_targets(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Return leakage-safe features, numeric G3, and performance categories.

    Categories are defined as Low: G3 0-9, Medium: 10-14, High: 15-20.
    """
    if TARGET not in frame.columns:
        raise ValueError(f"Required target column {TARGET!r} is missing")
    features = frame.drop(columns=[c for c in LEAKAGE_COLUMNS if c in frame.columns]).copy()
    regression_target = pd.to_numeric(frame[TARGET], errors="raise").astype(int)
    classification_target = pd.cut(
        regression_target,
        bins=[-1, 9, 14, 20],
        labels=["Low", "Medium", "High"],
    )
    if classification_target.isna().any():
        raise ValueError("G3 values must be between 0 and 20 to create performance categories")
    return features, regression_target, classification_target.astype(str)
