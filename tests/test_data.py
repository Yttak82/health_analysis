from src.data import count_missing_values, check_unique_ids, validate_data, validate_column_type
import pandas as pd


def test_count_missing_values(test_data):
    result = count_missing_values(test_data)

    expected ={
        "id": 1,
        "age":1,
        "sex": 1,
        "height": 1,
        "weight": 1,
        "systolic_bp": 1,
        "smoker": 1,
        "disease":0,
        "test": 4
    }

    assert result.to_dict() == expected


def  test_check_unique_ids(test_data):
    """Check wether all ids are unique."""
    result = check_unique_ids(test_data)

    assert result is False
    

def test_validate_data(test_data):
    """Check whether all values i column belong to allowed categories"""
    sex = validate_data(test_data, "sex", ["F","M"])
    smoker = validate_data(test_data, "smoker", ["Yes","No"])
    disease = validate_data(test_data, "disease", [0, 1])

    assert sex is False
    assert smoker is False
    assert disease is True


def test_validate_column_type(test_data):
        id = validate_column_type(test_data, "id", "int")
        age = validate_column_type(test_data, "age", "int")
        sex = validate_column_type(test_data, "sex", "srt")
        height = validate_column_type(test_data, "height", "float")
        weight = validate_column_type(test_data, "weight", "float")
        systolic_bp = validate_column_type(test_data, "systolic_bp", "float")
        smoker = validate_column_type(test_data, "smoker","srt")
        disease = validate_column_type(test_data, "disease", "int")
       
        assert id is False
        assert age is False
        assert sex is False
        assert height is True
        assert weight is False
        assert systolic_bp is True
        assert smoker is False
        assert disease is True