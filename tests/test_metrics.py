import pytest
import numpy as np
import pandas as pd
from src.metrics import prepare_regression_data, calculate_multiple_regression, calculate_pca, calculate_metrics, calculate_histogram, calculate_smoking_sex_interaction


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

def test_calculate_pca():
    """Test PCA dimensions, missing values and explained variance."""

    df = pd.DataFrame({
        "age": [20, 30, 40, 50, np.nan],
        "weight": [60, 65, 75, 90, 100],
        "systolic_bp": [110, 120, 130, 145, 160]
    })

    columns = ["age", "weight", "systolic_bp"]

    pca, X_pca, used_index = calculate_pca(df, columns)

    assert X_pca.shape == (4, 3)
    assert used_index.tolist() == [0, 1, 2, 3]
    assert np.isclose(
        pca.explained_variance_ratio_.sum(),
        1.0)
    assert np.isfinite(X_pca).all()


def test_prepare_regression_data():
    """Test category encoding and missing-value handling."""

    df = pd.DataFrame({
        "age": [20, 30, 40, 50],
        "weight": [60, 75, np.nan, 90],
        "sex": ["F", "M", "F", "M"],
        "smoker": ["No", "Yes", "Yes", "No"],
        "systolic_bp": [110, 130, 140, 150]
    })

    predictors = ["age", "weight", "sex", "smoker"]

    X, y = prepare_regression_data(df, predictors)

    assert X.shape == (3, 4)
    assert y.shape == (3,)
    assert X.index.tolist() == [0, 1, 3]
    assert X["sex"].tolist() == [0, 1, 1]
    assert X["smoker"].tolist() == [0, 1, 0]
    assert y.tolist() == [110, 130, 150]
    assert X.columns.tolist() == predictors


def test_prepare_regression_data_invalid_category():
    """Reject unexpected categorical values in regression data."""

    df = pd.DataFrame({
        "age": [25, 40, 55],
        "sex": ["F", "M", "Unknown"],
        "systolic_bp": [120, 135, 150]
    })

    with pytest.raises(ValueError, match="Unexpected category in sex"):
        prepare_regression_data( df,["age", "sex"])


def test_calculate_multiple_regression():
    """Test regression coefficients using known linear data."""

    age = np.array([20, 30, 40, 50, 60, 70])
    weight = np.array([60, 80, 65, 90, 70, 100])

    bp = 5 + 0.5 * age + 0.8 * weight

    df = pd.DataFrame({
        "age": age,
        "weight": weight,
        "systolic_bp": bp
    })

    model, X, y = calculate_multiple_regression(
        df,
        ["age", "weight"]
    )

    assert X.shape == (6, 2)
    assert y.shape == (6,)
    assert np.allclose(model.coef_, [0.5, 0.8])
    assert np.isclose(model.intercept_, 5.0)
    assert np.isclose(model.score(X, y), 1.0)
    assert np.allclose(model.predict(X), y)