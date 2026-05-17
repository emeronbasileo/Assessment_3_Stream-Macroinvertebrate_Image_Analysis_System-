'''
**************************************************************
Author:
u3323255   Assessment 3_Group 4_Pipeline Runner   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""Command-line pipeline runner - index, EDA, train, save the model, print results."""

import logging

from src import config
from src.services.workflow_service import WorkflowService


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    metrics = WorkflowService().run_full_pipeline()
    print()
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Train samples: {metrics['n_train']}    Test samples: {metrics['n_test']}")
    print(f"Features: {metrics['n_features']}    Classes: {metrics['n_classes']}")
    print()
    print((config.OUTPUTS_REPORTS_DIR / "classification_report.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
