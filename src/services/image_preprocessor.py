'''
**************************************************************
Author:
u3334531   Assessment 3_Group 4_Image Preprocessor   May 2026
Programming:
Python 3.12
**************************************************************
'''
"""ImagePreprocessor - load → grayscale → resize → normalise → flatten.

Adapted from ST1 Week 5 tutorial (NumPy & OpenCV foundations):
- cv2.imread for image loading
- cv2.cvtColor(..., COLOR_BGR2GRAY) for grayscale conversion
- cv2.resize for spatial normalisation

Adapted from ST1 Week 6 tutorial (Keras image handling activity, Task1_5):
- pixel normalisation by dividing the array by 255.0

Modified to: chain these into a single feature-extraction pipeline returning a
one-dimensional float32 vector suitable for an sklearn RandomForest, rather
than a Keras tensor or a batched ImageDataGenerator.
"""

from pathlib import Path

import cv2
import numpy as np

from src import config


cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)


class ImagePreprocessor:
    """Convert one image file to a flattened, normalised feature vector."""

    def __init__(self, image_size: tuple[int, int] = config.IMAGE_SIZE) -> None:
        """Construct the preprocessor with a target ``image_size`` (width, height)."""
        self.image_size: tuple[int, int] = image_size

    def load_features(self, image_path: Path) -> np.ndarray:
        """Read an image, preprocess it, and return its flattened feature vector.

        Args:
            image_path (Path): Path to the image file to process.

        Returns:
            np.ndarray: 1-D float32 array of length ``image_size[0] * image_size[1]``,
                with values normalised to [0, 1].

        Raises:
            FileNotFoundError: If ``image_path`` cannot be decoded by OpenCV.
        """
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Could not decode image: {image_path}")
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, self.image_size)
        normalised = resized.astype(np.float32) / 255.0
        return normalised.flatten()
