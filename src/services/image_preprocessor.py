import cv2
import numpy as np

class ImagePreprocessor:
    """Convert raw images into model-ready numeric features."""

    def __init__(self, image_size: tuple[int, int] = (128, 128)) -> None:
        self.image_size = image_size

    def transform(self, file_path: str) -> np.ndarray:
        """Load, resize, normalize, and flatten one image."""

        image = cv2.imread(str(file_path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {file_path}")

        resized = cv2.resize(image, self.image_size)
        normalized = resized.astype("float32") / 255.0
        return normalized.flatten()