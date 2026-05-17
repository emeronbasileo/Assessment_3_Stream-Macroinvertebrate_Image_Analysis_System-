'''
**************************************************************
Author:
u3323255   Assessment 3_Group 4_Tkinter GUI   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""MacroApp - Tkinter GUI for image classification (primary deployment path).

Adapted from ST1 Week 7 tutorial (MobileNetV2 transfer learning notebook,
predict_new_images function)
"""

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

from src import config
from src.services.workflow_service import WorkflowService


class MacroApp:
    """Tkinter app: pick an image, predict its species, display confidence."""

    PREVIEW_SIZE: tuple[int, int] = (400, 300)
    WINDOW_GEOMETRY: str = "620x560"
    SUPPORTED_FILETYPES: tuple[tuple[str, str], ...] = (
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("All files", "*.*"),
    )

    def __init__(self, workflow: WorkflowService | None = None) -> None:
        self.workflow: WorkflowService = workflow or WorkflowService()
        self.image_path: Path | None = None
        self._photo_ref: ImageTk.PhotoImage | None = None  # GC pin for the preview image

        self.root: tk.Tk = tk.Tk()
        self.root.title("Stream Macroinvertebrate Classifier")
        self.root.geometry(self.WINDOW_GEOMETRY)
        self.root.resizable(False, False)
        self._build_ui()

    def run(self) -> None:
        """Start the Tkinter main loop. Returns when the user closes the window."""
        self.root.mainloop()

    def _build_ui(self) -> None:
        title = ttk.Label(
            self.root,
            text="Stream Macroinvertebrate Classifier",
            font=("Segoe UI", 14, "bold"),
        )
        title.pack(pady=(18, 8))

        subtitle = ttk.Label(
            self.root,
            text="Choose an image of a macroinvertebrate to classify its species.",
            font=("Segoe UI", 10),
        )
        subtitle.pack(pady=(0, 12))

        self.preview_frame: ttk.Frame = ttk.Frame(
            self.root,
            width=self.PREVIEW_SIZE[0],
            height=self.PREVIEW_SIZE[1],
            relief="solid",
            borderwidth=1,
        )
        self.preview_frame.pack(pady=10)
        self.preview_frame.pack_propagate(False)

        self.preview_label: ttk.Label = ttk.Label(
            self.preview_frame,
            text="No image selected",
            font=("Segoe UI", 10),
        )
        self.preview_label.pack(expand=True)

        self.status_label: ttk.Label = ttk.Label(
            self.root,
            text="Choose an image to begin.",
            font=("Segoe UI", 11),
        )
        self.status_label.pack(pady=14)

        button_frame = ttk.Frame(self.root)
        button_frame.pack(pady=10)

        ttk.Button(
            button_frame, text="Choose Image", command=self._choose_image, width=16
        ).pack(side="left", padx=10)
        ttk.Button(
            button_frame, text="Predict", command=self._predict, width=16
        ).pack(side="left", padx=10)

    def _choose_image(self) -> None:
        path_str = filedialog.askopenfilename(
            title="Select an image", filetypes=self.SUPPORTED_FILETYPES
        )
        if not path_str:
            return

        path = Path(path_str)
        if path.suffix.lower() not in config.SUPPORTED_EXTENSIONS:
            messagebox.showerror(
                "Unsupported file type",
                f"File extension {path.suffix!r} is not supported.\n"
                f"Choose one of: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}",
            )
            return

        try:
            image = Image.open(path)
            image.thumbnail(self.PREVIEW_SIZE)
            self._photo_ref = ImageTk.PhotoImage(image)
        except Exception as exc:
            messagebox.showerror("Cannot open image", str(exc))
            return

        self.image_path = path
        self.preview_label.configure(image=self._photo_ref, text="")
        self.status_label.configure(text=f"Loaded: {path.name}")

    def _predict(self) -> None:
        if self.image_path is None:
            messagebox.showwarning(
                "No image selected",
                "Please choose an image first using 'Choose Image'.",
            )
            return
        try:
            label, confidence = self.workflow.predict_image(self.image_path)
        except FileNotFoundError as exc:
            messagebox.showerror(
                "Cannot predict",
                f"{exc}\n\nTrain the classifier first "
                f"(run 'python -m src.main' or use the console app).",
            )
            return
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return

        self.status_label.configure(
            text=f"Predicted: {label}  ({confidence * 100:.2f}% confidence)"
        )


def main() -> None:
    MacroApp().run()


if __name__ == "__main__":
    main()
