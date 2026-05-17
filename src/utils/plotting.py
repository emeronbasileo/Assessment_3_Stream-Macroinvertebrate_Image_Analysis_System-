'''
**************************************************************
Author:
u3318477   Assessment 3_Group 4_Plotting Helpers   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""Shared plotting helpers - theme setup, palette constants, sample-grid renderer.

Imported by every chart-producing service so visual style is configured in one place
and stays consistent across all the charts we produce.
"""

import math
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


PRIMARY_COLOR: str = "#2C7FB8"
SECONDARY_COLOR: str = "#D95F02"
DPI: int = 150


def apply_theme() -> None:
    """Configure seaborn and matplotlib for clean, presentation-quality charts.

    Safe to call more than once.
    """
    sns.set_theme(style="whitegrid", context="notebook", font_scale=1.0)
    plt.rcParams["figure.dpi"] = DPI
    plt.rcParams["savefig.dpi"] = DPI
    plt.rcParams["savefig.bbox"] = "tight"
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["figure.titleweight"] = "bold"


def render_sample_grid(
    image_paths_by_label: dict[str, Path],
    output_path: Path,
    suptitle: str,
    figsize: tuple[int, int] = (10, 10),
    cols: int = 3,
) -> None:
    """Render an N-image grid (one image per label) and save it to ``output_path``.

    Images are read with cv2 (BGR) and converted to RGB for matplotlib display.
    """
    n = len(image_paths_by_label)
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = np.atleast_1d(axes).ravel()

    for ax, (label, path) in zip(axes, image_paths_by_label.items()):
        bgr = cv2.imread(str(path))
        if bgr is None:
            ax.text(0.5, 0.5, "Image unavailable", ha="center", va="center")
            ax.set_axis_off()
            continue
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        ax.imshow(rgb)
        ax.set_title(label, fontsize=10)
        ax.set_axis_off()

    for ax in axes[len(image_paths_by_label):]:
        ax.set_axis_off()

    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=DPI)
    plt.close(fig)
