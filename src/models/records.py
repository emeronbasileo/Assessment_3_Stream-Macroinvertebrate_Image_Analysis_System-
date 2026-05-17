'''
**************************************************************
Author:
u3318477   Assessment 3_Group 4_Image Record Dataclass   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""Dataclass definitions for indexed dataset records."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """One indexed image: its file path, class label, and pixel dimensions.

    Once captured by the indexer, a record cannot be changed - it represents
    what was on disk at the time of indexing.
    """

    file_path: Path
    label: str
    width: int
    height: int
    channels: int
