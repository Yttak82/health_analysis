import pandas as pd
import pytest

@pytest.fixture
def test_data():
    return pd.DataFrame({
        "id": [1, None, 2, 3 , 3],
        "age": [15, 80, None, 50, 65],
        "sex": ["F", "M", None, "M", "k"],
        "height": [180, None, 200, 150, 170],
        "weight": ["60", 50, 60, 70, None],
        "systolic_bp": [None, 140, 150, 160, 180],
        "smoker": ["No", "Yes", "No", "No", None],
        "disease": [1, 0, 1, 1, 0],
        "test": [None, None, None, 2, None]

    })