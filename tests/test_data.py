import pandas as pd
import pytest

from student_performance.data import prepare_targets


def sample_frame():
    return pd.DataFrame({
        "school": ["GP", "MS", "GP"], "age": [16, 17, 18],
        "G1": [10, 12, 14], "G2": [11, 13, 16], "G3": [9, 12, 18],
    })


def test_prepare_targets_excludes_grade_leakage_and_bands_grades():
    X, regression, classification = prepare_targets(sample_frame())
    assert list(X.columns) == ["school", "age"]
    assert regression.tolist() == [9, 12, 18]
    assert classification.tolist() == ["Low", "Medium", "High"]


def test_prepare_targets_rejects_out_of_range_grade():
    frame = sample_frame()
    frame.loc[0, "G3"] = 21
    with pytest.raises(ValueError, match="between 0 and 20"):
        prepare_targets(frame)


def test_prepare_targets_requires_g3():
    with pytest.raises(ValueError, match="G3"):
        prepare_targets(pd.DataFrame({"age": [16]}))
