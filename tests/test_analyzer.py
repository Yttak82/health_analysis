import pytest 
from src.analyzer import HealthAnalyzer


def test_summary(test_data):
    analyzer = HealthAnalyzer(test_data)

    result = analyzer.summary("age")

    assert result["count"] == 4
    assert result["mean"] == pytest.approx(52.5)
    assert result["median"] == pytest.approx(57.5)
    assert result["min"] == 15
    assert result["max"] == 80
    assert result["std"] == pytest.approx(27.8388218141)
    assert result["se"] == pytest.approx(13.9194109070)


