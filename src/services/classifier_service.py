'''
**************************************************************
Author:
u3334531   Assessment 3_Group 4_Classifier Service   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""ClassifierService - feature/label matrix assembly, RandomForest training, evaluation, persistence."""

import logging
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from src import config
from src.services.image_preprocessor import ImagePreprocessor
from src.utils import plotting


LOGGER = logging.getLogger(__name__)


class ClassifierService:
    """Train, evaluate, and persist the baseline RandomForest classifier.

    Delegates feature extraction to ``ImagePreprocessor`` and writes evaluation
    artefacts (classification report + confusion matrix) to ``outputs/reports/``.
    """

    def __init__(
        self,
        preprocessor: ImagePreprocessor | None = None,
        n_estimators: int = config.N_ESTIMATORS,
    ) -> None:
        """Construct the service with an optional preprocessor and tree count."""
        self.preprocessor: ImagePreprocessor = preprocessor or ImagePreprocessor()
        self.n_estimators: int = n_estimators

    def build_xy(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        """Build the feature matrix and label vector from an indexed DataFrame.

        Args:
            df (pd.DataFrame): Indexed dataset, must contain ``file_path`` and
                ``label`` columns.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(X, y)`` where X is shape
                ``(n_samples, n_features)`` float32 and y is shape
                ``(n_samples,)`` string labels.
        """
        feature_rows: list[np.ndarray] = []
        labels: list[str] = []
        for _, row in df.iterrows():
            feature_rows.append(self.preprocessor.load_features(row["file_path"]))
            labels.append(row["label"])
        return np.vstack(feature_rows), np.array(labels)

    def train(
        self,
        df: pd.DataFrame,
        test_size: float = config.TEST_SIZE,
        save_model: bool = True,
        save_reports: bool = True,
    ) -> dict[str, Any]:
        """Fit the RandomForest on a stratified split and optionally persist artefacts.

        Args:
            df (pd.DataFrame): Indexed dataset.
            test_size (float): Fraction of samples held out for evaluation.
            save_model (bool): If True, save the trained model to ``config.MODEL_PATH``.
            save_reports (bool): If True, write the classification report and the
                confusion matrix into ``config.OUTPUTS_REPORTS_DIR``.

        Returns:
            dict[str, Any]: Metrics with keys ``accuracy``, ``n_train``,
                ``n_test``, ``n_features``, ``n_classes``.
        """
        X, y = self.build_xy(df)
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            stratify=y,
            random_state=config.RANDOM_SEED,
        )

        model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            random_state=config.RANDOM_SEED,
            n_jobs=-1,
        )
        LOGGER.info(
            "Training RandomForest: %d train samples, %d features, %d trees",
            X_train.shape[0],
            X_train.shape[1],
            self.n_estimators,
        )
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)

        if save_reports:
            self._write_reports(
                y_test=y_test,
                y_pred=y_pred,
                accuracy=accuracy,
                report=report,
                classes=sorted(np.unique(y).tolist()),
            )

        if save_model:
            joblib.dump(model, config.MODEL_PATH)
            LOGGER.info("Saved model to %s", config.MODEL_PATH)

        return {
            "accuracy": float(accuracy),
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "n_features": int(X_train.shape[1]),
            "n_classes": int(len(np.unique(y))),
        }

    @staticmethod
    def load_model() -> RandomForestClassifier:
        """Load the persisted classifier from ``config.MODEL_PATH``.

        Returns:
            RandomForestClassifier: The trained model loaded from disk.

        Raises:
            FileNotFoundError: If no model file exists at ``config.MODEL_PATH``.
        """
        if not config.MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"No trained model found at {config.MODEL_PATH}. Train first."
            )
        return joblib.load(config.MODEL_PATH)

    @staticmethod
    def _write_reports(
        y_test: np.ndarray,
        y_pred: np.ndarray,
        accuracy: float,
        report: str,
        classes: list[str],
    ) -> None:
        report_path = config.OUTPUTS_REPORTS_DIR / "classification_report.txt"
        report_path.write_text(
            f"Accuracy: {accuracy:.4f}\n\n{report}\n",
            encoding="utf-8",
        )

        plotting.apply_theme()
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=classes,
            yticklabels=classes,
            ax=ax,
            cbar=False,
        )
        ax.set_xlabel("Predicted class")
        ax.set_ylabel("True class")
        ax.set_title("Confusion matrix on the held-out test set")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        plt.setp(ax.get_yticklabels(), rotation=0)
        fig.tight_layout()
        fig.savefig(
            config.OUTPUTS_REPORTS_DIR / "confusion_matrix.png", dpi=plotting.DPI
        )
        plt.close(fig)
