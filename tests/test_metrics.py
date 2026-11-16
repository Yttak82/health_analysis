import pytest
import numpy as np
from src.metrics import calculate_metrics


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

 