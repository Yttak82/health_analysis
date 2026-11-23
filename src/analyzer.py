from src.metrics import calculate_simple_regression, calculate_power_simulation, calculate_welch_test, calculate_bootstrap_ci, calculate_bootstrap_stability, calculate_bootstrap_means, calculate_normal_ci, calculate_disease_difference, calculate_disease_rate, simulate_disease_group, calculate_smoking_stats, calculate_weight_by_sex, calculate_metrics, calculate_histogram, calculate_smoker_bp_stats, calculate_smoking_sex_interaction,  calculate_multiple_regression, calculate_regression_by_sex, calculate_regression_with_cholesterol
from src.viz import plot_age_vs_bp, plot_weight_vs_bp
from src.config import SEX_COLUMN, SMOKER_COLUMN, EXPECTED_DATATYPES
from src.data import check_unique_ids, validate_data, count_missing_values, validate_column_type


class HealthAnalyzer:
    """Perform statistical analysis of health data."""

    def __init__(self, df):
        """Initialize the analyzer with a health dataset."""
        self.df =df

    def summary(self, column):
        """Calculate descriptive statistick for a selected column."""
        return calculate_metrics(self.df, column)  
    
    def validate(self):
        "Validate dataset IDs and verifying ecpected categorical values"
        return{
        "id": check_unique_ids(self.df),
        "sex": validate_data(self.df, "sex", SEX_COLUMN),
        "smoker": validate_data(self.df, "smoker", SMOKER_COLUMN),
        "disease": validate_data(self.df, "disease", [0, 1]),
        }

    def missing_values(self):
        """count missing values in each dataset column"""
        return count_missing_values(self.df)
    
    def validate_types(self):
        """Validate dataset columns against expected data types."""
        return {
        column: "Korrekt" if validate_column_type(self.df, column, expected_type) 
        else "Fel" for column, expected_type in EXPECTED_DATATYPES.items()
        }

    
    def histogram(self, column, bins):
        "Calculate histogram counts, percantages and bin centers"
        values = self.df[column].dropna()
        return calculate_histogram(values, bins)

    def weight_by_sex(self, sex_column):
        """Calculate weight distributions and median by sex."""
        return calculate_weight_by_sex(self.df, sex_column)

    def smoking_stats(self, sex_column, smoker_column):
        """Calculate smoking counts and percentages by sex"""
        return calculate_smoking_stats(self.df, sex_column, smoker_column)

    def disease_rate(self):
        """Calculate the proportion of individuals with disease"""
        return calculate_disease_rate(self.df)

    def simulate_disease(self, group_size, seed):
        """Simulate diseare prevalence in a new group"""
        return simulate_disease_group(self.disease_rate(), group_size, seed)

    def disease_difference(self, simulated_rate):
        """Calculate the difference between simulated and observed disease prevalence."""
        return calculate_disease_difference(self.disease_rate(), simulated_rate)

    def normal_ci(self, column, alpha):
        """Calculate a normal-approximation confidence interval for the mean."""
        stats= self.summary(column)

        return calculate_normal_ci(
            stats["mean"],
            stats["se"],
            alpha
        )

    def bootstrap_means(self, column, n_bootstrap, batch_size, seed):
        """Generate bootstrap sample means for a selected column."""
        values = self.df[column].dropna()

        return calculate_bootstrap_means(
            values, 
            n_bootstrap,
            batch_size,
            seed
        )

    def welch_test(self):
        """Perform Welch´s t-test comparing smokers and non-smokers."""
        smoker_bp, no_smoker_bp, *_ = self.smoking_analysis()

        return calculate_welch_test(smoker_bp, no_smoker_bp)

    def power_simulation(self, sample_sizes, n_power_sim, alpha, seed, batch_size=250):
        """Estimate statistical power using simulated Welch´s t-tests."""

        (smoker_bp, non_smoker_bp,
         smoker_n, non_smoker_n,
         smoker_mean, no_smoker_mean,
         smoker_std, non_smoker_std,
         smoker_fraction,) = self.smoking_analysis()

        return calculate_power_simulation(
            smoker_fraction=smoker_fraction,
            smoker_mean=smoker_mean,
            smoker_std=smoker_std,
            non_smoker_mean=no_smoker_mean,
            non_smoker_std=non_smoker_std,
            sample_sizes=sample_sizes,
            n_power_sim=n_power_sim,
            alpha=alpha,
            seed=seed,
            batch_size=batch_size,
        )
    def bootstrap_stability(self, bootstrap_means, alpha, start, n_bootstrap, step):
        """Evaluate confidence interval stability across bootstrap sample sizes."""
        return calculate_bootstrap_stability(bootstrap_means, alpha, start, n_bootstrap, step)
    
    def bootstrap_ci(self, bootstrap_means, alpha):
        """Calculate a percentile bootstrap confidence interval."""
        return calculate_bootstrap_ci(bootstrap_means, alpha)
    
    def smoking_analysis(self):
        """Calculate blood pressure statistics for smokers and non_smokers."""
        return calculate_smoker_bp_stats(self.df)

    def simple_regression(self, column):
        "Fit  a simple linear regression for a selected predictor."
        return calculate_simple_regression(self.df, column)
    
    def multiple_regression(self):
        "Fit multiple linear reggression for systolic blood pressure."
        return calculate_multiple_regression(self.df)

    def regression_by_sex(self):
        """Fit a regression model including sex as an explanatory variabel"""
        return calculate_regression_by_sex(self.df)

    def regression_with_cholesterol(self):
        """Fit a regression model including sex and cholesterol as explanatory variabels"""
        return calculate_regression_with_cholesterol(self.df)

    def smoking_sex_interaction(self):
        """Test whether the smoking-BP association differs by sex."""
        return calculate_smoking_sex_interaction(self.df)

    def plot_age_vs_bp(self):
        """Visualiseze the relationship between age and systolic blood pressure."""
        return plot_age_vs_bp(self.df)

    def plot_weight_vs_bp(self):
        """Visualiseze the relationship between weight and systolic blood pressure."""
        return plot_weight_vs_bp(self.df)