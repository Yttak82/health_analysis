import pytest
import numpy as np
import pandas as pd
from src.metrics import calculate_metrics, calculate_histogram, calculate_smoking_sex_interaction


def test_calculate_metrics(test_data):
    result = calculate_metrics(test_data, "age")

    assert result["count"] == 4
    assert result["mean"] == pytest.approx(52.5)
    assert result["median"] == pytest.approx(57.5)
    assert result["min"] == 15
    assert result["max"] == 80
    assert result["std"] == pytest.approx(27.8388218141)
    assert result["se"] == pytest.approx(13.9194109070)


def test_single_value(test_data):
    result = calculate_metrics(test_data, "test")

    assert result["count"] == 1
    assert result["mean"] == 2
    assert np.isnan(result["std"])
    assert np.isnan(result["se"])


def test_calculate_histogram():
    values = np.array([120,125,130,140])
    bins = np.array([120, 130, 140, 150])

    counts, percentages, centers = calculate_histogram(values, bins)

    assert counts.tolist() == [2, 1, 1]
    assert percentages.tolist() == [50.0, 25.0, 25.0]
    assert centers.tolist() == [125.0, 135.0, 145.0]

def test_smoking_sex_interaction():
    df = pd.DataFrame({
        "age": [20, 35, 55, 61, 35, 17, 53, 85],
        "sex": ["F", "F", "M", "M", "M", "M", "F", "F"],
        "weight": [60, 70, 80, 90, 120, 75, 45, 95],
        "systolic_bp":[ 110, 135, 140, 155, 125, 138, 145, 160],
        "smoker": ["No", "Yes", "No", "Yes", "No", "Yes", "No", "Yes"],
        })
    model = calculate_smoking_sex_interaction(df)

    assert "smoker_male" in model.params.index
    assert 0 <= model.pvalues["smoker_male"]<= 1