'''
**************************************************************
Author:
u3318477   Assessment 3_Group 4_Configuration   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""Project-wide constants: paths, image-processing parameters, supported extensions.

Imported by every service so paths and tunables live in one place. Path resolution is
anchored to this file's location, so the constants stay correct regardless of which
entry point (main.py, app.py, console_app.py) starts the process or what the working
directory happens to be.
"""

from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

DATA_DIR: Path = PROJECT_ROOT / "data"
DATA_RAW_DIR: Path = DATA_DIR / "raw"

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
OUTPUTS_EDA_DIR: Path = OUTPUTS_DIR / "eda"
OUTPUTS_MODELS_DIR: Path = OUTPUTS_DIR / "models"
OUTPUTS_REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

MODEL_PATH: Path = OUTPUTS_MODELS_DIR / "macro_classifier.joblib"

IMAGE_SIZE: tuple[int, int] = (128, 128)

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".bmp"})

RANDOM_SEED: int = 42

# Classifier defaults (callers may override via constructor / method args).
N_ESTIMATORS: int = 100
TEST_SIZE: float = 0.2

# EDA chart defaults.
HISTOGRAM_BINS: int = 20
