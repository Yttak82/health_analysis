from src.metrics import calculate_metrics


class HealthAnalyzer:
    """Perform health data analysis."""

    def __init__(self, df):
        self.df =df
    def summary(self, column):
        """Descriptive statistick for a column."""
        return calculate_metrics(self.df, column)    