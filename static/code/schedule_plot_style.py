"""Shared Matplotlib style from the SGD shared-warmup reference figure.

Curve colors describe their role, not the optimizer: optimal/best-found is
charcoal, linear orange, cosine blue, and constant taupe. All axes are linear.
"""

import os
from pathlib import Path
from tempfile import gettempdir

INK, MUTED, GRID = "#383832", "#74746c", "#e7e8e2"
LINEAR, COSINE, CONSTANT = "#b56b30", "#517b8b", "#94826d"
WARMUP_FILL, WARMUP_EDGE = "#efefeb", "#c0c3b9"


def configure(hash_salt):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(gettempdir()) / "schedule-plot-mpl"))
    os.environ.setdefault("XDG_CACHE_HOME", str(Path(gettempdir()) / "schedule-plot-cache"))
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager

    font = Path(__file__).resolve().parents[1] / "fonts/source-serif-4/SourceSerif4-Regular.ttf"
    family = "DejaVu Serif"
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        family = font_manager.FontProperties(fname=str(font)).get_name()
    matplotlib.rcParams.update({
        "font.family": family, "font.size": 11, "mathtext.fontset": "stix",
        "text.color": INK, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.edgecolor": "#c1c4bb", "axes.linewidth": .8,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.dpi": 160,
        "svg.fonttype": "path", "svg.hashsalt": hash_salt,
    })
