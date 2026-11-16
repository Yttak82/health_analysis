import numpy as np

def calculate_metrics(df, column):
    """Calculate descriptive statistics for a column."""

    values = df[column].dropna()

    count = len(values)
    mean = values.mean()
    median = values.median()
    max = values.max()
    min = values.min()
    std = values.std(ddof=1)
    
    if count > 1:
        se = std / np.sqrt(count)
    else:
        se = np.nan

    return {
        "count": count,
        "mean": mean,
        "median": median,
        "max": max,
        "min": min,
        "std": std,
        "se": se,
    }