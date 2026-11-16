def count_missing_values(df):
    """Count missing values in each DataFrame column."""
    return df.isna().sum()


def  check_unique_ids(df):
    """Check wether all ids are unique."""
    return bool(df["id"].is_unique)


def validate_data(df, column, allowed_values):
    """Check whether all values i column belong to allowed categories"""
    return bool(df[column].isin(allowed_values).all())


def validate_column_type(df, column, expected_type):
    """check wheteher all datatyper i column have expected_type"""
    return all(
        type(value).__name__ == expected_type
        for value in df[column]
    )
