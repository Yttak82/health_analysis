import matplotlib.pyplot as plt
import numpy as np
from src.styles import style_axis

def plot_histogram(
    ax,
    bin_centers,
    values,
    bin_width,
    color,
    ylabel,
    title,
    mean,
    median,
):
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


def plot_boxplot(
    ax,
    weight_groups,
    weight_labels,
    sex_column,
    sex_colors,
    colors,
    median_weight_by_sex,
):
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

    for patch, sex in zip(weight_box["boxes"], sex_column):
        patch.set(
            facecolor=sex_colors[sex], 
                                alpha=0.6,
                                edgecolor=sex_colors[sex],
                                linewidth=1.5,
        )
    for pos, sex in enumerate(sex_column, start=1):
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
               
def plot_smoking_status(ax,
                        smoking_all_counts,
                        smoker_column,
                        smoking_colors,
                        colors):
     smoking_labels = {
          "No": "Icke-rökare",
          "Yes": "Rökare",
     }
     total = smoking_all_counts.sum()

     for idx, status in enumerate(smoker_column):
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
               range(len(smoker_column)),
               [smoking_labels[status] for status in smoker_column],
          )

          ax.invert_yaxis()

          ax.set(
                    title="Rökstatus i hela datasetet",
                    xlabel="Antal personer",
               )


def plot_smoking_by_sex(
          axes, 
          smoking_counts,
          smoking_perc,
          sex_column,
          smoker_column,
          smoking_colors,
          colors,
):
     smoking_labels = {
          "No": "Icke-rökare",
          "Yes": "Rökare",
     }
     sex_names = ["Kvinnor", "Män"]
     sex_positions = np.arange(len(sex_column))

     for axis, table, subtitle, xlabel, is_percent in [
          (axes[0], smoking_counts, "Antal", "Antal personer", False),
          (axes[1], smoking_perc, "Andel inom kön", "Andel (%)", True)
     ]:
          left = np.zeros(len(sex_column))

          for status in smoker_column:
               widths = table[status].to_numpy()

               bars = axis.barh(
                    sex_positions,
                    widths,
                    left=left,
                    height=0.52,

                 label=smoking_labels[status],
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


def plot_bootstrap_stability(
              ax,
              bootstrap_check_sizes,
              bootstrap_ci_changes,
              stop_at,
              colors
    ):
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


def plot_age_vs_bp(df):
      "Plot the ralationship between age and systolic blood pressure."
      fig, ax = plt.subplots(figsize=(9, 5))

      ax.scatter(
            df["age"],
            df["systolic_bp"],
            alpha=0.4,
            color="steelblue"
      )

      ax.set_title("Samband mellan ålder och systoliskt blodtryck")
      ax.set_xlabel("Ålder (år)")
      ax.set_ylabel("Systoliskt blodtryck (mmHg)")
      ax.grid(alpha=0.3)

      plt.tight_layout()
      plt.show()


def plot_weight_vs_bp(df):
      "Plot the ralationship between age and systolic blood pressure."
      fig, ax = plt.subplots(figsize=(9, 5))

      ax.scatter(
            df["weight"],
            df["systolic_bp"],
            alpha=0.4,
            color="steelblue"
      )

      ax.set_title("Samband mellan vikt och systoliskt blodtryck")
      ax.set_xlabel("Vikt (kg)")
      ax.set_ylabel("Systoliskt blodtryck (mmHg)")
      ax.grid(alpha=0.3)

      plt.tight_layout()
      plt.show()

def plot_cholesterol_vs_bp(df):
      "Plot the ralationship between cholesterol and systolic blood pressure."
      fig, ax = plt.subplots(figsize=(9, 5))

      ax.scatter(
            df["cholesterol"],
            df["systolic_bp"],
            alpha=0.4,
            color="steelblue"
      )

      ax.set_title("Samband mellan cholesterol och systoliskt blodtryck")
      ax.set_xlabel("Cholesterol ()")
      ax.set_ylabel("Systoliskt blodtryck (mmHg)")
      ax.grid(alpha=0.3)

      plt.tight_layout()
      plt.show()

def plot_regression_predictions(y_true, y_pred):
              
    fig, ax = plt.subplots(figsize=(7,6)
    )
    ax.scatter(
        y_true, y_pred, alpha=0.4, color="steelblue")
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

    data = df[[column, "systolic_bp"]].dropna()

    if categories is None:
                ax.scatter(
                    data[column],
                    data["systolic_bp"],
                    alpha=0.4,
                    color="steelblue"
                    )
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

                labels = {"F":"Kvinnor",
                        "M":"Män",
                        "No":"Icke-rökare",
                        "Yes":"Rökare"}

                ax.set_xticklabels([labels.get(category, category) for category in categories]
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
            (axes[2, 0], "sex", "Kön", ["F", "M"], "Kön"),
            (axes[2, 1], "smoker", "Rökstatus",["No", "Yes"], "Rökstatus"),
      ]

      for ax, column, xlabel, categories, title in plots:
            plot_bp_relationship( ax, df, column, xlabel, categories=categories)
            ax.set_title(title)

      fig.suptitle("Samband med systoliskt blootryck",
                   fontsize=16,
                   fontweight="bold")

      return fig, axes