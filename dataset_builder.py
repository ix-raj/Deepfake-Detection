import argparse
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from feature_utils import extract_power_spectrum_features, iter_image_files


def load_folder_dataset(dataset_root: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    class_map = {"real": 0, "fake": 1}
    features: list[np.ndarray] = []
    labels: list[int] = []
    counts: dict[str, int] = {}

    for class_name, class_label in class_map.items():
        class_folder = dataset_root / class_name
        if not class_folder.exists():
            raise FileNotFoundError(f"Missing required folder: {class_folder}")

        image_paths = iter_image_files(class_folder)
        counts[class_name] = len(image_paths)

        for image_path in image_paths:
            with Image.open(image_path) as image:
                feature_vector, _, _ = extract_power_spectrum_features(image.convert("RGB"))
            features.append(feature_vector)
            labels.append(class_label)

    if not features:
        raise ValueError(f"No supported images found under {dataset_root}")

    X = np.vstack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int32)
    metadata = {
        "dataset_root": str(dataset_root),
        "class_distribution": counts,
        "sample_count": int(len(labels)),
        "feature_count": int(X.shape[1]),
    }
    return X, y, metadata


def main():
    parser = argparse.ArgumentParser(
        description="Build a power-spectrum dataset from folders named real/ and fake/."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        required=True,
        help="Folder containing real/ and fake/ image directories.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("power_spectrum_dataset.pkl"),
        help="Output pickle path.",
    )
    args = parser.parse_args()

    X, y, metadata = load_folder_dataset(args.dataset_root)
    dataset = {"power_spectrum": X, "label": y, "metadata": metadata}

    with args.output.open("wb") as file:
        pickle.dump(dataset, file)

    print(f"Saved dataset to {args.output}")
    print("Samples:", metadata["sample_count"])
    print("Features per sample:", metadata["feature_count"])
    print("Class distribution:", metadata["class_distribution"])


if __name__ == "__main__":
    main()
