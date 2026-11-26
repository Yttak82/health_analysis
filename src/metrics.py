import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from src.config import BP_COLUMN, BINARY_MAPPINGS


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


def calculate_histogram(values, bins):
    """Calculate histogram counts and percentages."""
    counts, _ = np.histogram(values, bins=bins)
    percentages = counts / len(values) * 100
    centers = (bins[:-1] + bins[1:]) / 2

    return counts, percentages, centers


def calculate_weight_by_sex(df, sex_categories):
    """Prepere weight groups and median weights by sex."""
    
    weight_groups = [
        df.loc[df["sex"] == sex, "weight"].dropna() for sex in sex_categories]
    median_weight_by_sex = df.groupby("sex")["weight"].median()

    return weight_groups, median_weight_by_sex


def calculate_smoking_stats(df, sex_categories, smoking_categories):
    """Calculate smoking counts and percentages."""

    smoking_counts = pd.crosstab(df["sex"],
                                df["smoker"],
                                ).reindex(
                                    index=sex_categories,
                                    columns=smoking_categories,
                                    fill_value=0
                                )
    smoking_perc =(
        smoking_counts.div(
            smoking_counts.sum(axis=1),
            axis=0
        ) * 100
    )

    smoking_all_counts = smoking_counts.sum(axis=0)

    return smoking_counts, smoking_perc, smoking_all_counts


def calculate_disease_rate(df):
    """Calculate the proportion of people with disease."""
   
    return df["disease"].mean()


def simulate_disease_group(disease_rate, group_size, seed):
    """Simulate disasease status for a group using a binomial disytribution.
        Arg: disease_rate - probability of disease in the population
        Returns : Proportion of simulated people with disease"""
    rng_disease = np.random.default_rng(seed)
    simulated_objects = rng_disease.binomial(1, disease_rate, size=group_size)
    
    return simulated_objects.mean()


def calculate_disease_difference(disease_rate, simulated_rate):    
    """Calculate the differense between simulated ond observed disease rates.
    Args:   disease_rate: Observed disease proportion
            simulated_rate: simulated disease proportion"""
    return simulated_rate - disease_rate


def calculate_normal_ci(bp_mean, bp_se, alpha):
    """Calculate a confidence interval using normal approximation"""
    normal_margin = stats.norm.ppf(1 - alpha / 2) * bp_se
    normal_ci = (bp_mean -normal_margin, bp_mean + normal_margin) 
    
    return normal_margin, normal_ci


def calculate_bootstrap_means(bp_values, n_bootstrap, batch_size, seed):
    """Generate bootstrap sample means from blood pressure values.
        Args:   bp_values:Orginal blood pressure observations
                n_bootstrap: number of bootstrap per batch
                batch_size: Number of samples processed per batch """
    rng_bootstrap =np.random.default_rng(seed)
    bp_array = bp_values.to_numpy()
    bootstrap_means= np.empty(n_bootstrap, dtype=float)
    bootstrap_batch_size = batch_size
    
    for batch_start in range(0, n_bootstrap, bootstrap_batch_size):
        batch_end = min(batch_start + bootstrap_batch_size, n_bootstrap)
        samples = rng_bootstrap.choice(bp_array, size=(batch_end - batch_start, len(bp_array)), replace=True)
        bootstrap_means[batch_start:batch_end] = samples.mean(axis=1)
    
    return bootstrap_means


def calculate_bootstrap_ci(bootstrap_means, alpha):
    """Calculate bootstrap confidence interval."""
    
    return tuple(np.percentile(bootstrap_means, [100 * alpha / 2, 100 * (1 - alpha / 2)]))


def calculate_bootstrap_stability(bootstrap_means, alpha, bootstrap_start, n_bootstrap, bootstrap_step):
    """Evaluate the stability of bootstrap confidence intervals.
        Args: bootstrap_means - Previously calculated bootstrap means.
            alpha - Significance level used for the confidence interval.
            bootstrap_start - Initial number of bootstrap samples.
            n_bootstrap - Maximum number of bootstrap samples.
            bootstrap_step - Increase in sample size between checks.
        Returns:
            Tuple containing sample sizes, lower and upper confidence
            bounds, changes between consecutive intervals, and the
            first sample size meeting the stability criterion.
            The final value is None if stability is not reached.
    """

    bootstrap_check_sizes = list(range(bootstrap_start, n_bootstrap, bootstrap_step))

    if not bootstrap_check_sizes or bootstrap_check_sizes[-1] != n_bootstrap:
        bootstrap_check_sizes.append(n_bootstrap)
    bootstrap_lower_bounds = []
    bootstrap_upper_bounds = []
    bootstrap_ci_changes = []
    previous_ci = None

    for n_replicates in bootstrap_check_sizes:
        current_ci = tuple(np.percentile(bootstrap_means[:n_replicates],
                                    [100 * alpha / 2, 100 * (1 - alpha / 2)]))
        bootstrap_lower_bounds.append(current_ci[0])
        bootstrap_upper_bounds.append(current_ci[1])
        if previous_ci is not None:
            bootstrap_ci_changes.append(max(abs(current_ci[0] - previous_ci[0]),
                                        abs(current_ci[1] - previous_ci[1])))
        previous_ci = current_ci

    tolerance = 0.01
    checks_needed = 10
    stop_at = None

    for i in range(checks_needed, len(bootstrap_ci_changes) + 1):
        recent_changes = bootstrap_ci_changes[i - checks_needed:i]

        if all(change <= tolerance for change in recent_changes):
            stop_at = bootstrap_check_sizes[i]
            break
    
    return (
    bootstrap_check_sizes,
    bootstrap_lower_bounds,
    bootstrap_upper_bounds,
    bootstrap_ci_changes,
    stop_at
)


def calculate_smoker_bp_stats(df):
    """Calculate blood pressure statistics for smokers and non-smokers"""
    smoker_bp = df[df["smoker"] == "Yes"]["systolic_bp"].dropna()
    no_smoker_bp = df[df["smoker"] == "No"]["systolic_bp"].dropna()

    smoker_n, non_smoker_n = len(smoker_bp), len(no_smoker_bp)
    smoker_mean, non_smoker_mean = smoker_bp.mean(), no_smoker_bp.mean()
    smoker_std, non_smoker_std = smoker_bp.std(ddof=1), no_smoker_bp.std(ddof=1)
    smoker_fraction = smoker_n / (smoker_n + non_smoker_n)

    return (
        smoker_bp, no_smoker_bp,
        smoker_n, non_smoker_n,
        smoker_mean, non_smoker_mean,
        smoker_std, non_smoker_std,
        smoker_fraction,
    )


def calculate_welch_test(smoker_bp, no_smoker_bp):
    """One-sided Welch t-test."""
    return stats.ttest_ind(smoker_bp, no_smoker_bp, equal_var=False, alternative="greater")


def calculate_power_simulation(smoker_fraction, smoker_mean, smoker_std, non_smoker_mean, non_smoker_std, sample_sizes, n_power_sim,
        alpha, seed, batch_size=250,):
        """Estimate statistical power using simulated Welch t-tests.

        Args: smoker_fraction - Proportion of smokers in the sample.
            sample_sizes - Total sample sizes to evaluate.
            n_power_sim - Number of simulations per sample size.
            alpha - Significance level for rejecting the null hypothesis.
            batch_size - Number of simulations processed per batch.
        Returns:
            DataFrame containing sample sizes, group sizes, and
            estimated statistical power for the one-sided Welch test.
        """
        rng_power = np.random.default_rng(seed)
        power_rows = []
        power_batch_size = batch_size

        for total_n in sample_sizes:
            simulated_smoker_n = round(total_n * smoker_fraction)
            simulated_non_smoker_n = total_n - simulated_smoker_n
            welch_significant = 0

            for batch_start in range(0, n_power_sim, power_batch_size):
                batch_n = min(power_batch_size, n_power_sim - batch_start)
                sim_smoker_power =rng_power.normal(
                    smoker_mean, smoker_std, size=(batch_n, simulated_smoker_n)
                )
                sim_no_smoker_power =rng_power.normal(
                    non_smoker_mean, non_smoker_std, size=(batch_n, simulated_non_smoker_n)
                )
                wech_p = stats.ttest_ind(
                    sim_smoker_power, sim_no_smoker_power, axis=1, equal_var=False, alternative="greater").pvalue
                welch_significant += np.count_nonzero(wech_p < alpha)

            power_rows.append({
                "Total antal:": total_n,
                "Simulation - antal rökare:": simulated_smoker_n,
                "Simulation - antal icke-rökare:": simulated_non_smoker_n,
                "Welch power:": welch_significant / n_power_sim,
            })

        return pd.DataFrame(power_rows)


def prepare_regression_data(df, predictors):
    """Prepare predictor and target data for linear regression."""
    columns = list(predictors) + [BP_COLUMN]
    data = df[columns].dropna().copy()

    for column in predictors:
        if column in BINARY_MAPPINGS:
            data[column]  = data[column].map(BINARY_MAPPINGS[column])
            if data[column].isna().any():raise ValueError(f"Unexpected category in {column}")

    X = data[list(predictors)]
    y = data[BP_COLUMN]

    return X, y


def calculate_simple_regression(df, column):
    "Fit a simple linear regression for a numeric predictor."

    X, y = prepare_regression_data(df, [column])
   
    model =LinearRegression()
    model.fit(X, y)
    
    return model, X, y


def calculate_multiple_regression(df, predictors=None):
    """Fit a multiple linear regression using selected predictors."""

    if predictors is None:
        predictors = ["age", "weight", "smoker"]

    X, y = prepare_regression_data(df, predictors)
    X = X.to_numpy()
    y = y.to_numpy()

    model = LinearRegression()
    model.fit(X, y)

    return model, X, y


def calculate_regression_inference(df, predictors, alpha=0.05):
    """Calculate regression coefficients, p-values and confidence intervals using OLS"""
    X, y = prepare_regression_data(df, predictors)

    X = sm.add_constant(X.astype(float))
    model = sm.OLS(y.astype(float),X).fit()

    ci = model.conf_int(alpha=alpha)

    results = pd.DataFrame({
        "Koefficient": model.params,
        "Standartfel": model.bse,
        "t-värde": model.tvalues,
        "p-värde": model.pvalues,
        "Nedre KI": ci[0],
        "Övre KI": ci[1],
        "Signifikant": np.select([model.pvalues < alpha,
        model.pvalues >=alpha],
        ["Ja", "Nej"], 
        default="Ej bedömbart")
        })
    
    return model, results

def calculate_regression_by_sex(df):
    """Fit a multiple linear regression model for systolic blood pressure inkluding sex and smoking status."""
    predictors = ["age", "weight", "smoker", "sex"]
    
    return calculate_multiple_regression(df, predictors)


def calculate_regression_with_cholesterol(df):
    """Fit a multiple linear regression model for systolic blood pressure inkluding sex and cholesterol."""
    predictors = ["age", "weight", "smoker", "sex", "cholesterol"]
    
    return calculate_multiple_regression(df, predictors)


def calculate_smoking_sex_interaction(df):
    """Test whether the smoking-BP association differs by sex."""
    import statsmodels.formula.api as smf
    data = df[["age","weight","smoker","sex","systolic_bp"]].dropna().copy()
    data["smoker_num"] = data["smoker"].map(BINARY_MAPPINGS["smoker"])
    data["male"] = data["sex"].map(BINARY_MAPPINGS["sex"])
    data["smoker_male"] = data["smoker_num"]*data["male"]

    model =smf.ols("systolic_bp ~ age +weight +smoker_num + male+ smoker_male", data=data).fit()

    return model


def calculate_standardized_coefficients(model, X, y):
    """Calculate standardized regression coefficients."""

    x_std = np.std(X, axis=0)
    y_std =np.std(y)

    if y_std == 0 or np.any(x_std == 0):

        raise ValueError("Cannot standardize constant variabled")

    return model.coef_ * x_std / y_std


def evaluate_regression_model(X, y, test_size=0.2, seed=42):
    """Evaluate linear regression using training and test data."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=seed
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    return {
        "Train R^2": model.score(X_train, y_train),
          "Test R^2": model.score(X_test, y_test), 
          "MAE": mean_absolute_error(y_test, y_pred),
          "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
          "Train antal": len(y_train), "Test antal": len(y_test)}


def calculate_regression_cv(X, y, n_splits=5, seed=42):
    """Evaluate linear regression using K-fold cross-validation."""

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(LinearRegression(), X, y, cv=kfold, scoring={"r2":"r2", "mae": "neg_mean_absolute_error","rmse": "neg_root_mean_squared_error"})

    return {"CV R^2": scores["test_r2"].mean(),
            "CV R^2 std": scores["test_r2"].std(),
            "CV MAE":-scores["test_mae"].mean(),
            "CV RMSE": -scores["test_rmse"].mean()}


def calculate_pca(df, columns):
    """Perform PCA on selected numerical variables."""

    data = df[columns].dropna().copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(data)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    return pca, X_pca, data.index