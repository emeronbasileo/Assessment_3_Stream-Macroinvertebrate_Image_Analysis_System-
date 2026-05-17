'''
**************************************************************
Author:
u3323255   Assessment 3_Group 4_Console Application   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""ConsoleApp - menu-driven console for running the pipeline (backup demo path).

Adapted from ST1 Week 7 tutorial (MobileNetV2 transfer learning notebook,
predict_new_images function):

Modified to: run interactively from a console rather than iterating a fixed
test directory.
"""

from pathlib import Path

from src import config
from src.services.workflow_service import WorkflowService


class ConsoleApp:
    """Backup demo path: menu-driven console for the macro_project pipeline."""

    MENU: str = """
==============================================
  Stream Macroinvertebrate Classifier - Menu
==============================================
  1. Show dataset summary
  2. Generate EDA outputs
  3. Train baseline classifier
  4. Predict an image
  5. Exit
"""

    def __init__(self, workflow: WorkflowService | None = None) -> None:
        self.workflow: WorkflowService = workflow or WorkflowService()

    def run(self) -> None:
        """Display the menu and dispatch user choices until Exit."""
        actions = {
            "1": self._show_summary,
            "2": self._run_eda,
            "3": self._train,
            "4": self._predict,
        }
        while True:
            print(self.MENU)
            try:
                choice = input("Select an option: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye.")
                return

            if choice == "5":
                print("Goodbye.")
                return

            action = actions.get(choice)
            if action is None:
                print(f"Invalid option: {choice!r}. Please choose 1-5.")
                continue

            try:
                action()
            except Exception as exc:
                print(f"Error: {exc}")

    def _show_summary(self) -> None:
        df = self.workflow.index_dataset()
        print(f"\nTotal images:  {len(df)}")
        print(f"Total classes: {df['label'].nunique()}")
        print(
            f"Width  range: {df['width'].min()}-{df['width'].max()}, "
            f"mean {df['width'].mean():.1f}"
        )
        print(
            f"Height range: {df['height'].min()}-{df['height'].max()}, "
            f"mean {df['height'].mean():.1f}"
        )
        print("\nClass counts (descending):")
        for label, count in df["label"].value_counts().items():
            print(f"  {label:<24s} {count}")

    def _run_eda(self) -> None:
        print("Running EDA - indexing dataset and generating charts...")
        self.workflow.run_eda()
        print(f"EDA artefacts saved to: {config.OUTPUTS_EDA_DIR}")

    def _train(self) -> None:
        print("Training classifier - this takes approx 1-2min")
        metrics = self.workflow.train_classifier()
        print(f"Accuracy:      {metrics['accuracy']:.4f}")
        print(f"Train samples: {metrics['n_train']}")
        print(f"Test samples:  {metrics['n_test']}")
        print(f"Model saved to: {config.MODEL_PATH}")

    def _predict(self) -> None:
        path_str = input("Enter image path: ").strip().strip('"')
        if not path_str:
            print("No path entered.")
            return
        label, confidence = self.workflow.predict_image(Path(path_str))
        print(f"Predicted: {label}  ({confidence * 100:.2f}% confidence)")


def main() -> None:
    ConsoleApp().run()


if __name__ == "__main__":
    main()
