'''
**************************************************************
Author:
u3334531   Assessment 3_Group 4_Workflow Service   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""WorkflowService - connects the indexer, EDA, classifier, and preprocessor.

Called by main.py, app.py, and console_app.py so all three run the same code
when asked to run EDA, train the classifier, or predict an image.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src import config
from src.services.classifier_service import ClassifierService
from src.services.dataset_indexer import DatasetIndexer
from src.services.eda_service import EDAService
from src.services.image_preprocessor import ImagePreprocessor


class WorkflowService:
    """Used by every entry point to run the pipeline.

    Creates the ``outputs/`` subdirectories on construction and caches the indexed
    DataFrame so later calls do not re-scan the disk.
    """

    def __init__(self) -> None:
        """Ensure output directories exist and initialise an empty dataset cache."""
        for directory in (
            config.OUTPUTS_DIR,
            config.OUTPUTS_EDA_DIR,
            config.OUTPUTS_MODELS_DIR,
            config.OUTPUTS_REPORTS_DIR,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        self._cached_df: pd.DataFrame | None = None

    def _get_dataset(self) -> pd.DataFrame:
        """Index the dataset on first call; return the cached DataFrame thereafter."""
        if self._cached_df is None:
            self._cached_df = DatasetIndexer().index()
        return self._cached_df

    def index_dataset(self) -> pd.DataFrame:
        """Return the indexed dataset DataFrame, scanning lazily on first call.

        Returns:
            pd.DataFrame: The indexed dataset, cached for the lifetime of this service.
        """
        return self._get_dataset()

    def run_eda(self) -> pd.DataFrame:
        """Generate every Stage 1 EDA artefact and return the indexed DataFrame.

        Returns:
            pd.DataFrame: The indexed DataFrame, returned for caller reuse.
        """
        df = self._get_dataset()
        EDAService(df).run_all()
        return df

    def train_classifier(self) -> dict[str, Any]:
        """Train the baseline RandomForest and persist model + reports.

        Returns:
            dict[str, Any]: Training metrics from ``ClassifierService.train``
                (accuracy, n_train, n_test, n_features, n_classes).
        """
        df = self._get_dataset()
        return ClassifierService().train(df)

    def run_full_pipeline(self) -> dict[str, Any]:
        """Index once, then EDA + train in sequence.

        Returns:
            dict[str, Any]: Training metrics from ``ClassifierService.train``.
        """
        df = self._get_dataset()
        EDAService(df).run_all()
        return ClassifierService().train(df)

    def predict_image(self, image_path: Path) -> tuple[str, float]:
        """Predict the species class for a single image.

        Args:
            image_path (Path): Path to the image to classify.

        Returns:
            tuple[str, float]: ``(label, confidence)`` where confidence is in [0, 1].
        """
        features = ImagePreprocessor().load_features(image_path).reshape(1, -1)
        model = ClassifierService.load_model()
        probabilities = model.predict_proba(features)[0]
        best = int(np.argmax(probabilities))
        return str(model.classes_[best]), float(probabilities[best])
