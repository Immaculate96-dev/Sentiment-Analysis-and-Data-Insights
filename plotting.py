"""Shared chart colours and matplotlib style used by every module."""

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

COLORS = {
    "positive": "#2E8B6E",
    "negative": "#B23A48",
    "neutral": "#7C8AA0",
    "accent": "#14305A",        # navy
    "accent_light": "#A9B8D0",  # soft navy
    "gold": "#C9A227",
    "grid": "#E1E6EE",
    "text": "#0B1F3A",
}

# white -> navy for counts and shares; gold <- white -> navy for correlations
NAVY_CMAP = LinearSegmentedColormap.from_list("navy", ["#FFFFFF", COLORS["accent"]])
NAVY_GOLD_CMAP = LinearSegmentedColormap.from_list("navy_gold", [COLORS["gold"], "#FFFFFF", COLORS["accent"]])

_POSITIVE_NAMES = {"positive", "pos", "1", "1.0", "true", "yes", "good", "y", "+", "p"}
_NEGATIVE_NAMES = {"negative", "neg", "0", "0.0", "false", "no", "bad", "n", "-"}


def apply_style():
    """Set a clean, consistent look for all charts."""
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": COLORS["grid"],
            "axes.labelcolor": COLORS["text"],
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.titlelocation": "left",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.8,
            "axes.axisbelow": True,
            "xtick.color": COLORS["text"],
            "ytick.color": COLORS["text"],
            "text.color": COLORS["text"],
            "font.size": 10,
            "figure.dpi": 100,
        }
    )


def color_for_label(label):
    """Green for positive-looking labels, red for negative-looking, blue otherwise."""
    name = str(label).strip().lower()
    if name in _POSITIVE_NAMES:
        return COLORS["positive"]
    if name in _NEGATIVE_NAMES:
        return COLORS["negative"]
    if name == "neutral":
        return COLORS["neutral"]
    return COLORS["accent"]


def new_axes(ax, figsize=(6, 3.6)):
    """Return (fig, ax), creating a new figure only if no axes was supplied."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        return fig, ax
    return ax.figure, ax
