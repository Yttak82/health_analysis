import matplotlib.pyplot as plt
import numpy as np
from src.styles import style_axis, SEX_COLORS, SMOKING_COLORS, COLORS
from src.config import SEX_LABELS, SMOKER_LABELS, SEX_COLUMN, SMOKER_COLUMN


def plot_histogram(ax, bin_centers, values, bin_width, color, ylabel, title, mean, median,):
    "Plot a blood pressure histograf with summary statistics, inkludes reference lines for the mean and median, a shaded blood pressure interval"
    ax.bar(
        bin_centers,
        values,
        width=bin_width * 0.90,
        color=color,
    )

    ax.set_title(title, loc="left")
    ax.set(
        xlabel="Systoliskt blodtryck (mmHg)",
        ylabel=ylabel,
    )

    ax.axvline(
            mean,
            color="black",
            linestyle="-.",
            linewidth=2,
            alpha=0.7,
            label=f"Medielvärde {mean:.2f} mmHg",
        )
    ax.axvline(
        median,
        color="green",
        linestyle="-",
        linewidth=2,
        alpha=0.7,
        label=f"Median {median:.2f} mmHg"
    )

    ax.axvspan(135,
               165,
               color="gray",
               alpha=0.3
    )
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3)


def plot_boxplot(ax, weight_groups, weight_labels, sex_categories, sex_colors, colors, median_weight_by_sex,):
    """Plot weight distributions by sex using boxplots"""
    weight_box = ax.boxplot(
        weight_groups,
        patch_artist=True,
        tick_labels=weight_labels,
        widths=0.55,
        medianprops={
            "color": colors["ink"],
            "linewidth":2.5,
        },
        capprops={
            "linewidth": 1,
            "markersize": 7,
            "alpha": 0.45,
            "markeredgecolor":colors["muted"]
        },
    )

    for patch, sex in zip(weight_box["boxes"], sex_categories):
        patch.set(
            facecolor=sex_colors[sex], 
                                alpha=0.6,
                                edgecolor=sex_colors[sex],
                                linewidth=1.5,
        )
    for pos, sex in enumerate(sex_categories, start=1):
            median = median_weight_by_sex.loc[sex]

            ax.annotate(
                 f"Median {median:.2f} kg",
                 (pos, median),
                 xytext=(75,-5),
                 textcoords="offset points",
                 fontsize=9,
                 color=colors["ink"],
                 fontweight="bold",
            )

            ax.set(
                 title="Viktfördelning efter kön",
                 xlabel="Kön",
                 ylabel="Vikt (kg)",
            )

            ax.set_yticks(range(0, 131, 10))

               
def plot_smoking_status(ax, smoking_all_counts, smoking_categories, smoking_colors, colors):
     """Plot the distribution of smoking status in the dataset - displays counts and percentages for smokers and non-smokers.
        Arg: smoking_all_counts - number of people in each smoking category"""
     
     total = smoking_all_counts.sum()

     for idx, status in enumerate(smoking_categories):
          count = int(smoking_all_counts.loc[status])
          ax.barh(
               idx,
               count,
               color=smoking_colors[status],
               height=0.6,
          )
          ax.text(count - 10,
                  idx,
                  f"{count} ({count / total:.2%})",
                  ha="right",
                  va="center",
                  fontweight="bold",
                  color=colors["text"],
                  fontsize=7,
          )

          ax.set_yticks(
               range(len(smoking_categories)),
               [SMOKER_LABELS[status] for status in smoking_categories],
          )

          ax.invert_yaxis()

          ax.set(
                    title="Rökstatus i hela datasetet",
                    xlabel="Antal personer",
               )


def plot_smoking_by_sex(axes, smoking_counts, smoking_perc, sex_categories, smoking_categories, smoking_colors, colors,):
     """Plot smoking status distributions separetly by sex, two charts show absolute counts and percentages within each sex category"""
     
     sex_names = [SEX_LABELS[sex] for sex in sex_categories]
     sex_positions = np.arange(len(sex_categories))

     for axis, table, subtitle, xlabel, is_percent in [
          (axes[0], smoking_counts, "Antal", "Antal personer", False),
          (axes[1], smoking_perc, "Andel inom kön", "Andel (%)", True)
     ]:
          left = np.zeros(len(sex_categories))

          for status in smoking_categories:
               widths = table[status].to_numpy()

               bars = axis.barh(
                    sex_positions,
                    widths,
                    left=left,
                    height=0.52,

                 label=SMOKER_LABELS[status],
                 color=smoking_colors[status],
               )
               for bar, value in zip(bars, widths):
                                   axis.text(
                                        bar.get_x() + bar.get_width() / 2,
                                        bar.get_y() + bar.get_height() / 2,
                               f"{value:.2f} %" if is_percent else f"{int(value)}",
                               ha="center",
                               va="center",
                               color=colors["text"],
                               fontweight="bold",
                               fontsize=8,
                               ) 
               left += widths

               axis.set_yticks(sex_positions, sex_names)
               axis.invert_yaxis()
               axis.set(title=subtitle, xlabel=xlabel,)
               axis.set_xlim(0, 105 if is_percent else float(table.sum(axis=1).max()) * 1.08)


def plot_bootstrap_stability(ax, bootstrap_check_sizes, bootstrap_ci_changes, stop_at, colors):
    """Vizualize the stability of bootstrap confidence intervals - shows chenges in interval bounds and marks the stability tolerance and suggested stopping point
        Args:bootstrap_check_sizes -  Numbers of bootstrap samples evaluated.
            bootstrap_ci_changes - Changes in confidence interval bounds between consecutive evaluations.
            stop_at - Suggested stopping point, or None if not reached.
        """
    ax.scatter(
        bootstrap_check_sizes[1:],
        bootstrap_ci_changes,
        color=colors["blood_pressure"]
    )

    ax.axhline(
        0.01,
        color=colors["reference"],
        linestyle="--",
        label="Tolerans: 0.01 mmHg"
    )

    ax.set(
        title="Stabilitet i bootstrapintervallets gränser",
        xlabel="Antal bootstrapreplikat",
        ylabel="Förändring av intervallgränserna (mmHg)"    
    )

    if stop_at is not None:
        ax.axvline(
            x=stop_at,
            color="red",
            linestyle="-.",
            linewidth=1.5,
            alpha=0.3,
            label=(f"Förslåg stopp: {stop_at:,}")
    )
    ax.legend()
    ax.set_ylim(bottom=-0.005)
    ax.set_xlim(left=0)

    style_axis(ax)


def plot_regression_predictions(y_true, y_pred):
    """Compare observed and predicted blood pressure values, includes a diagonal reference line
        representing perfect predictions.

        Args: y_true - Observed systolic blood pressure values.
              y_pred - Blood pressure values predicted by the model.
        """
    fig, ax = plt.subplots(figsize=(7,6))
    ax.scatter(
        y_true, y_pred, alpha=0.4, color=COLORS["pca_primary"])
    minimum = min(y_true.min(), y_pred.min())
    maximum = max(y_true.max(), y_pred.max())
        
    ax.plot(
        [minimum, maximum],
        [minimum, maximum],
        color="red",
        linestyle="--",
        label="Perfekt prediktion"
    )

    ax.set(
        title="Faktisk och predikterat systolisk blodtryck",
        xlabel="Faktisk blodtryck",
        ylabel="Predikterat blodtryck (mmHg)"    
    )

    
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_bp_relationship(ax, df, column, xlabel, categories=None):
    """Plot systolic blood pressure by a numeric och categorical variable."""

    data =df[[column, "systolic_bp"]].dropna()
    if categories is None:
                
                    
                x = data[column].to_numpy()
                y= data["systolic_bp"].to_numpy()
                ax.scatter( x, y,                
                                alpha=0.4,
                                color=COLORS["blood_pressure"]
                                )
                slope, intercept = np.polyfit(x, y, 1)
                x_line = np.linspace(x.min(), x.max(), 100)
                y_line = slope * x_line + intercept
                ax.plot(x_line, y_line, color=COLORS["regression_line"], linewidth=2, label="linjär regressionlinje"), 
                
    else:
                groups = [data.loc[
                    data[column] == category,
                    "systolic_bp"
                ]
                for category in categories
                ]

                ax.boxplot(
                    groups,
                    patch_artist=True,
                    widths=0.5
                )

                ax.set_xticks(range(1, len(categories) + 1))

                category_labels = {**SEX_LABELS,
                        **SMOKER_LABELS}

                ax.set_xticklabels([category_labels.get(category, category) for category in categories]
                                )
                
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Systoliskt blodtryck (mmHg)")
    ax.grid(alpha=0.5)

def plot_bp_overview(df):
      "Visualize six relationships with systolic blood pressure."

      fig, axes = plt.subplots(3, 2,
                               figsize=(25, 25),
                               sharey=True,
                               layout="constrained")

      plots = [
            (axes[0, 0], "age", "Ålder (år)", None, "Ålder"),
            (axes[0, 1], "height", "Längd (cm)", None, "Längd"),
            (axes[1, 0], "weight", "Vikt (kg)", None, "Vikt"),
            (axes[1, 1], "cholesterol", "Kolesterol", None, "Kolesterol"),
            (axes[2, 0], "sex", "Kön", SEX_COLUMN, "Kön"),
            (axes[2, 1], "smoker", "Rökstatus", SMOKER_COLUMN, "Rökstatus"),
      ]

      for ax, column, xlabel, categories, title in plots:
            plot_bp_relationship( ax, df, column, xlabel, categories=categories)
            ax.set_title(title)

      fig.suptitle("Samband med systoliskt blootryck",
                   fontsize=16,
                   fontweight="bold")

      return fig, axes

def plot_standardized_coefficients(coefficients):
      """Compare standardized coefficients across regression models."""
      fig, ax = plt.subplots(figsize=(10, 6))

      variables = list(coefficients.index)
      n_models = len(coefficients.columns)

      for i, model_name in enumerate(coefficients.columns):
          values = coefficients[model_name].dropna()

          offset = (i-(n_models - 1) / 2) * 0.22
          y_positions = [variables.index(variable) + offset for variable in values.index]

          ax.scatter(values, y_positions, s=70, label=model_name)

      labels = {
        "age": "Ålder",
        "cholesterol": "Kolesterol",
        "weight": "Vikt",
        "sex": "Kön",
        "smoker": "Rökstatus",
        "height": "Längd"
        }

      ax.set_yticks(range(len(variables)))
      ax.set_yticklabels([labels.get(variable, variable)
                            for variable in variables])
      ax.invert_yaxis()
      ax.axvline(0, color="gray", linestyle="--", alpha=0.7)

      ax.set_xlabel("Standardiserad regressionskoefficient (beta)")
      ax.set_title("Samband med systoliskt blodtryck\n"" Standardiserade koefficienter i tre regressionsmodeller.", fontweight="bold")
      ax.set_ylabel("Variabel")
      ax.grid(axis="x", alpha=0.3)
      ax.legend(title="Regressionsmodell", loc="lower right")
      fig.text(0.5, 0.02, "Varje punkt visar en variabels koefficient i en modell.\n""Höger om noll = positivt samband"   "Vänster om noll = negativt samband", ha="center", fontsize=9)
      fig.tight_layout(rect=(0, 0.12, 1, 1))

      return fig, ax

def plot_pca_variance(pca):
    """Plot explained variance of PCA components."""
    variance = pca.explained_variance_ratio_ * 100
    components = [f"PC{i + 1}" for i in range(len(variance))]

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.bar(components, variance, color=COLORS["pca_primary"])

    for i, value in enumerate(variance):
            ax.text(i, value + 0.5, f"{value:.1f}%", ha="center")

    ax.set_title("Förklarad varians per huvudkomponent")
    ax.set_xlabel("Huvudkomponent")
        
    ax.set_ylabel("Förklarad varians (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(variance) * 1.15)
                                    
        
    fig.tight_layout()
    return fig, ax

def plot_pca_scores(X_pca, pca):
    """Visualize observations using the first two PCA components."""

    fig, ax = plt.subplots(figsize=(9,6))

    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.5, s=25, color=COLORS["pca_primary"])

    variance = pca.explained_variance_ratio_ * 100
    ax.set_title("Observationer i PCA-rummet")

    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% förklarad varians)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% förklarad varians)")

    ax.axhline(0, color=COLORS["pca_reference"], alpha=0.4)
    ax.axvline(0, color=COLORS["pca_reference"], alpha=0.4)

    ax.grid(alpha=0.2)

    fig.tight_layout()

    return fig, ax


def plot_pca_by_category(X_pca, pca, group_values, category_order, labels, colors, title, legend_title):
    """Visualize PCA observations grouped by a categorical variable."""
    fig, ax = plt.subplots(figsize=(9,6))
    group_values = np.asarray(group_values)

    for category in category_order:
        mask = group_values == category

        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], alpha=0.5, s=25, color=colors[category], label=labels[category])
    
    variance = pca.explained_variance_ratio_ * 100
    ax.set_title(title)
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% förklarad varians)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% förklarad varians)")
    
    ax.axhline(0, color=COLORS["pca_reference"], alpha=0.4)
    ax.axvline(0, color=COLORS["pca_reference"], alpha=0.4)
    
    ax.grid(alpha=0.2)
    ax.legend(title=legend_title)
    fig.tight_layout()
    
    return fig, ax


def plot_pca_by_sex(X_pca, pca, sex):
    """Visualize PCA observations grouped by sex."""

    return plot_pca_by_category(
          X_pca=X_pca,
          pca=pca,
          group_values=sex,
          category_order=SEX_COLUMN,
          labels=SEX_LABELS,
          colors=SEX_COLORS,
          title="PCA - fördelning efter kön.",
          legend_title="Kön")

def plot_pca_by_smoker(X_pca, pca, smoker):
    """Visualize PCA observations grouped by rökningstatus."""

    return plot_pca_by_category(
              X_pca=X_pca,
              pca=pca,
              group_values=smoker,
              category_order=SMOKER_COLUMN,
              labels=SMOKER_LABELS,
              colors=SMOKING_COLORS,
              title="PCA - fördelning efter rökstatus.",
              legend_title="Rökstatus")


def plot_pca_biplot(X_pca, pca, columns):
    """Visualize PCA scores and variable directions."""
 
    fig, ax = plt.subplots(figsize=(10, 7))

    ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.25, s=20, color=COLORS["pca_primary"])

    loadings = pca.components_[:2, :].T
    scale = (0.65 * np.percentile(np.abs(X_pca[:, :2]), 95) / np.max(np.abs(loadings)))
    labels = {
            "age": "Ålder",
            "cholesterol": "Kolesterol",
            "weight": "Vikt",
            "systolic_bp": "Systoliskt blodtryck",
            "height": "Längd"
            }
    label_offsets = {
                        "age":(8, -12),
                        "cholesterol": (-15, 15),
                        "weight": (8, 12),
                        "systolic_bp": (15, -15),
                        "height": (-20, 15)
                        }
    for i, column in enumerate(columns):
        x = loadings[i, 0] * scale
        y = loadings[i, 1] * scale

        ax.annotate("", xy=(x, y),
                    xytext=(0, 0),
                    arrowprops=dict(arrowstyle="->", color=COLORS["pca_loading"], alpha=0.5, lw=2)
                    )
        ax.annotate(labels.get(column, column), xy=(x, y), xytext=label_offsets.get(column, (8, 8)), 
                    textcoords="offset points", color=COLORS["pca_loading"], 
                    fontsize=10, bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))
        
    variance = pca.explained_variance_ratio_ * 100
    ax.set_title("PCA - biplot - observationer och variabler")
 
    ax.set_xlabel(f"PC1 ({variance[0]:.1f}% förklarad varians)")
    ax.set_ylabel(f"PC2 ({variance[1]:.1f}% förklarad varians)")
 
    ax.axhline(0, color=COLORS["pca_reference"], alpha=0.4)
    ax.axvline(0, color=COLORS["pca_reference"], alpha=0.4)
 
    ax.grid(alpha=0.2)
    
    fig.tight_layout()
 
    return fig, ax
 
      