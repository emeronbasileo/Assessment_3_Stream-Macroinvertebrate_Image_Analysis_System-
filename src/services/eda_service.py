'''
**************************************************************
Author:
u3318477   Assessment 3_Group 4_Exploratory Data Analysis   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""EDAService - generates the Stage 1 charts and summary from an indexed dataset DataFrame."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from src import config
from src.utils import plotting


class EDAService:
    """Produce the four Stage 1 EDA artefacts (three charts and a summary text file)."""

    def __init__(self, df: pd.DataFrame, output_dir: Path | None = None) -> None:
        """Construct the EDA service for ``df``, writing artefacts to ``output_dir``."""
        self.df: pd.DataFrame = df
        self.output_dir: Path = (
            output_dir if output_dir is not None else config.OUTPUTS_EDA_DIR
        )

    def run_all(self) -> None:
        """Generate every Stage 1 EDA artefact in one call."""
        plotting.apply_theme()
        self.plot_class_distribution()
        self.plot_size_distributions()
        self.plot_sample_grid()
        self.write_summary_stats()

    def plot_class_distribution(self) -> None:
        """Write the class distribution bar chart to ``output_dir``."""
        counts = self.df["label"].value_counts().sort_values(ascending=False)

        fig, ax = plt.subplots(figsize=(12, 6))
        bars = ax.bar(range(len(counts)), counts.values, color=plotting.PRIMARY_COLOR)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels(counts.index, rotation=45, ha="right")
        ax.set_xlabel("Class")
        ax.set_ylabel("Number of images")
        ax.set_title("Number of images per class")

        for bar, value in zip(bars, counts.values):
            ax.annotate(
                str(int(value)),
                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )

        fig.tight_layout()
        fig.savefig(self.output_dir / "class_distribution.png", dpi=plotting.DPI)
        plt.close(fig)

    def plot_size_distributions(self) -> None:
        """Write the side-by-side width / height histogram chart to ``output_dir``."""
        fig, (ax_w, ax_h) = plt.subplots(1, 2, figsize=(12, 5))

        for ax, column, subtitle in (
            (ax_w, "width", "Image widths"),
            (ax_h, "height", "Image heights"),
        ):
            sns.histplot(
                self.df[column],
                bins=config.HISTOGRAM_BINS,
                ax=ax,
                color=plotting.PRIMARY_COLOR,
            )
            mean_value = self.df[column].mean()
            ax.axvline(
                mean_value,
                color=plotting.SECONDARY_COLOR,
                linestyle="--",
                linewidth=1.5,
                label=f"Mean = {mean_value:.0f} px",
            )
            ax.set_xlabel("Pixels")
            ax.set_ylabel("Number of images")
            ax.set_title(subtitle)
            ax.legend()

        fig.suptitle("Image dimension distributions", fontsize=14)
        fig.tight_layout()
        fig.savefig(self.output_dir / "image_size_distribution.png", dpi=plotting.DPI)
        plt.close(fig)

    def plot_sample_grid(self) -> None:
        """Write a 3x3 representative sample grid from the nine largest classes."""
        top_labels = (
            self.df["label"]
            .value_counts()
            .sort_values(ascending=False)
            .head(9)
            .index.tolist()
        )

        sample_by_label: dict[str, Path] = {}
        for label in top_labels:
            paths = self.df.loc[self.df["label"] == label, "file_path"].tolist()
            sample_by_label[label] = paths[len(paths) // 2]

        plotting.render_sample_grid(
            sample_by_label,
            self.output_dir / "sample_grid.png",
            suptitle="Representative samples from the nine largest classes",
            figsize=(10, 10),
            cols=3,
        )

    def write_summary_stats(self) -> None:
        """Write the dataset summary statistics text file to ``output_dir``."""
        df = self.df
        counts = df["label"].value_counts().sort_values(ascending=False)
        most_common_label = counts.index[0]
        least_common_label = counts.index[-1]
        imbalance = counts.iloc[0] / counts.iloc[-1]

        if df["channels"].nunique() == 1:
            channels_line = f"  Channels: all {df['channels'].iloc[0]}-channel"
        else:
            channels_line = f"  Channels: mixed {sorted(df['channels'].unique())}"

        lines = [
            "Stream Macroinvertebrate Dataset - Summary Statistics",
            "=" * 56,
            "",
            f"Total images:  {len(df)}",
            f"Total classes: {df['label'].nunique()}",
            "",
            "Image dimensions:",
            f"  Width  - min: {df['width'].min()}, max: {df['width'].max()}, mean: {df['width'].mean():.1f}",
            f"  Height - min: {df['height'].min()}, max: {df['height'].max()}, mean: {df['height'].mean():.1f}",
            channels_line,
            "",
            "Class distribution:",
            f"  Largest:  {most_common_label:<22s} {counts.iloc[0]} images",
            f"  Smallest: {least_common_label:<22s} {counts.iloc[-1]} images",
            f"  Imbalance ratio: {imbalance:.1f}x",
            "",
            "Per-class counts (descending):",
        ]
        for label, count in counts.items():
            lines.append(f"  {label:<24s} {count:>5d}")

        (self.output_dir / "summary_stats.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
