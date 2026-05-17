'''
**************************************************************
Author:
u3318477   Assessment 3_Group 4_Dataset Indexer   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""DatasetIndexer - recursively scan data/raw/<class>/ and build a DataFrame of records."""

import logging
from dataclasses import asdict
from pathlib import Path

import cv2
import pandas as pd

from src import config
from src.models.records import ImageRecord


LOGGER = logging.getLogger(__name__)


class DatasetIndexer:
    """Walk a class-foldered image dataset and emit a DataFrame of ImageRecord rows.

    Files with unsupported extensions or undecodable contents are skipped.
    """

    def __init__(self, root: Path | None = None) -> None:
        """Construct the indexer, defaulting to ``config.DATA_RAW_DIR`` when root is None."""
        self.root: Path = root if root is not None else config.DATA_RAW_DIR

    def index(self) -> pd.DataFrame:
        """Scan the dataset root and return one row per indexed image.

        Returns:
            pd.DataFrame: Columns file_path (Path), label (str), width (int),
                height (int), channels (int).

        Raises:
            FileNotFoundError: If the root directory is missing or contains
                no supported images.
        """
        if not self.root.is_dir():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")

        records: list[ImageRecord] = []
        for class_dir in sorted(p for p in self.root.iterdir() if p.is_dir()):
            label = class_dir.name
            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
                    continue
                record = self._read_record(image_path, label)
                if record is not None:
                    records.append(record)

        if not records:
            raise FileNotFoundError(
                f"No supported images found under {self.root}. "
                f"Download the Kaggle dataset from "
                f"https://www.kaggle.com/datasets/kennethtm/stream-macroinvertebrates "
                f"and unzip so the layout is data/raw/<class>/<image>."
            )

        return pd.DataFrame([asdict(r) for r in records])

    @staticmethod
    def _read_record(image_path: Path, label: str) -> ImageRecord | None:
        image = cv2.imread(str(image_path))
        if image is None:
            LOGGER.warning("Could not decode image, skipping: %s", image_path)
            return None
        height, width = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        return ImageRecord(
            file_path=image_path,
            label=label,
            width=width,
            height=height,
            channels=channels,
        )
