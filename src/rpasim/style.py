import matplotlib as mpl
import seaborn as sns

LINEPLOT_WIDTH = 2.5


def set_style():
    """Default style for matplotlib and seaborn plots."""
    sns.set_style("ticks")
    sns.set_palette("plasma")
    mpl.rcParams["pdf.fonttype"] = 42  # Nature compliance
