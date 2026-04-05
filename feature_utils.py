from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.interpolate import interp1d


TARGET_SIZE = (128, 128)
FEATURE_LENGTH = 722
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def normalize_image(image: Image.Image) -> np.ndarray:
    """Reduce simple quality differences before extracting FFT features."""
    gray = np.array(image.convert("L"))
    resized = cv2.resize(gray, TARGET_SIZE, interpolation=cv2.INTER_AREA)
    denoised = cv2.GaussianBlur(resized, (3, 3), 0)
    normalized = cv2.equalizeHist(denoised)
    return normalized


def extract_power_spectrum_features(image: Image.Image) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    normalized = normalize_image(image)

    # Focus on medium/high frequencies to reduce sensitivity to brightness.
    spectrum = np.fft.fftshift(np.fft.fft2(normalized))
    magnitude = np.log1p(np.abs(spectrum))
    magnitude[TARGET_SIZE[1] // 2 - 8: TARGET_SIZE[1] // 2 + 8,
              TARGET_SIZE[0] // 2 - 8: TARGET_SIZE[0] // 2 + 8] = 0

    flat_features = magnitude.flatten()
    x_old = np.linspace(0, 1, len(flat_features))
    x_new = np.linspace(0, 1, FEATURE_LENGTH)
    interpolator = interp1d(x_old, flat_features, kind="linear")
    features = interpolator(x_new).astype(np.float32)
    return features, magnitude, normalized


def iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        path for path in folder.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
