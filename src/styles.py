
import matplotlib.pyplot as plt

COLORS = {
    "blood_pressure": "#65758B",
    "blood_pressure_light": "#8CACDA",
    "non_smoker": "#5F4763",
    "smoker": "#AE703C",
    "female": "#52CC52",
    "male": "#E1D26D",
    "reference": "#153D1E",
    "reference_m": "#34EB71",
    "ink": "#202020",
    "muted": "#595959",
    "text": "#E5FAC4",
    "regression_line": "#371D24",
    "pca_primary": "#7AB3E2",
    "pca_loading": "#D85656",
    "pca_reference": "#808080"
}

SMOKING_COLORS = {
    "No": COLORS["non_smoker"], 
    "Yes": COLORS["smoker"]
    }

SEX_COLORS = {
    "F": COLORS["female"],
    "M": COLORS["male"]
    }


def setup_plot_style():
    """Set default appesrence for plots."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "figure.facecolor":  "#FFFFFF",
        "savefig.dpi": 150,

        "font.family":  "DejaVu Sans",
        "font.size": 10,
        "text.color": COLORS["ink"],
   
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.facecolor": "#FFFFFF",
        "axes.labelsize": 10,
        "axes.labelcolor": COLORS["muted"],
        "axes.edgecolor": "#9C9494",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.axisbelow": True,

        "xtick.color": COLORS["muted"],
        "ytick.color": COLORS["muted"],

        "grid.color": "#A59A9A9B",

        "legend.frameon": True,
        "legend.facecolor": "#FFFFFF",
        "legend.framealpha": 0.95,
        "legend.fontsize": 8,
        "legend.labelcolor": COLORS["muted"]
})

def style_axis(axis, grid_axis="y"):
    "Apply consistent grid and spine styling"
    axis.grid(axis=grid_axis, linewidth=0.5, alpha=0.5)
    axis.spines["left"].set_alpha(0.7)
    axis.spines["bottom"].set_alpha(0.7)