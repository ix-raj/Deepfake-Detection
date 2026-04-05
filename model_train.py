import argparse
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from dataset_builder import load_folder_dataset


DEFAULT_DATA_PATH = Path("power_spectrum_1000.pkl")
DEFAULT_MODEL_PATH = Path("model.pkl")
RANDOM_STATE = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and compare multiple deepfake classifiers."
    )
    parser.add_argument(
        "--data-pkl",
        type=Path,
        default=DEFAULT_DATA_PATH,
        help="Existing pickle dataset with power_spectrum and label arrays.",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        help="Folder with real/ and fake/ subfolders. If provided, features are built from images.",
    )
    parser.add_argument(
        "--cache-dataset",
        type=Path,
        help="Optional path to save the folder-built dataset as a pickle for future runs.",
    )
    parser.add_argument(
        "--output-model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to save the trained model bundle.",
    )
    return parser.parse_args()


def build_candidate_models() -> dict[str, Pipeline]:
    return {
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        max_iter=4000,
                        class_weight="balanced",
                        solver="liblinear",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "extra_trees": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=300,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=1,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "model",
                    HistGradientBoostingClassifier(
                        max_depth=8,
                        learning_rate=0.08,
                        max_iter=250,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, float]:
    best_threshold = 0.5
    best_score = -1.0

    for threshold in np.arange(0.30, 0.701, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        score = f1_score(y_true, y_pred, zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = float(round(threshold, 2))

    return best_threshold, best_score


def load_dataset(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, dict]:
    if args.dataset_root:
        X, y, metadata = load_folder_dataset(args.dataset_root)
        metadata["source"] = "folders"
        if args.cache_dataset:
            dataset = {"power_spectrum": X, "label": y, "metadata": metadata}
            with args.cache_dataset.open("wb") as file:
                pickle.dump(dataset, file)
            print(f"Cached extracted dataset to {args.cache_dataset}")
        return X, y, metadata

    with args.data_pkl.open("rb") as file:
        dataset = pickle.load(file)

    metadata = dataset.get("metadata", {})
    metadata["source"] = "pickle"
    metadata["dataset_path"] = str(args.data_pkl)
    return dataset["power_spectrum"], dataset["label"].astype(int), metadata


def evaluate_candidate(
    name: str,
    pipeline: Pipeline,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> dict:
    pipeline.fit(X_train, y_train)
    y_val_prob = pipeline.predict_proba(X_val)[:, 1]
    threshold, val_f1 = find_best_threshold(y_val, y_val_prob)
    val_pred = (y_val_prob >= threshold).astype(int)
    return {
        "name": name,
        "pipeline": pipeline,
        "threshold": threshold,
        "val_f1": float(val_f1),
        "val_auc": float(roc_auc_score(y_val, y_val_prob)),
        "val_accuracy": float(accuracy_score(y_val, val_pred)),
        "val_balanced_accuracy": float(balanced_accuracy_score(y_val, val_pred)),
    }


def main():
    args = parse_args()
    X, y, metadata = load_dataset(args)
    X, y = shuffle(X, y, random_state=RANDOM_STATE)

    # Keep the test split untouched until the final evaluation.
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Use a validation split for threshold tuning and model comparison.
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        stratify=y_train_full,
        random_state=RANDOM_STATE,
    )

    candidate_models = build_candidate_models()
    results = []

    for model_name, pipeline in candidate_models.items():
        try:
            result = evaluate_candidate(model_name, pipeline, X_train, y_train, X_val, y_val)
        except Exception as exc:
            print(f"{model_name}: skipped due to training error ({exc})")
            continue

        results.append(result)
        print(
            f"{model_name}: "
            f"val_f1={result['val_f1']:.4f}, "
            f"val_auc={result['val_auc']:.4f}, "
            f"threshold={result['threshold']:.2f}"
        )

    if not results:
        raise RuntimeError("All candidate models failed to train.")

    best_result = max(
        results,
        key=lambda item: (item["val_f1"], item["val_auc"], item["val_balanced_accuracy"]),
    )

    final_pipeline = build_candidate_models()[best_result["name"]]
    final_pipeline.fit(X_train_full, y_train_full)

    y_test_prob = final_pipeline.predict_proba(X_test)[:, 1]
    y_test_pred = (y_test_prob >= best_result["threshold"]).astype(int)

    print("\nBest model:", best_result["name"])
    print("Chosen threshold:", best_result["threshold"])
    print("Test Accuracy:", round(accuracy_score(y_test, y_test_pred), 4))
    print("Test Balanced Accuracy:", round(balanced_accuracy_score(y_test, y_test_pred), 4))
    print("Test AUC:", round(roc_auc_score(y_test, y_test_prob), 4))
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_test_pred, digits=4))

    leaderboard = [
        {
            "name": result["name"],
            "threshold": result["threshold"],
            "val_f1": round(result["val_f1"], 4),
            "val_auc": round(result["val_auc"], 4),
            "val_accuracy": round(result["val_accuracy"], 4),
            "val_balanced_accuracy": round(result["val_balanced_accuracy"], 4),
        }
        for result in sorted(results, key=lambda item: item["val_f1"], reverse=True)
    ]

    model_bundle = {
        "pipeline": final_pipeline,
        "threshold": best_result["threshold"],
        "metadata": {
            "dataset": metadata,
            "feature_count": int(X.shape[1]),
            "sample_count": int(X.shape[0]),
            "class_distribution": {
                "real": int(np.sum(y == 0)),
                "fake": int(np.sum(y == 1)),
            },
            "best_model": best_result["name"],
            "leaderboard": leaderboard,
            "test_metrics": {
                "accuracy": float(accuracy_score(y_test, y_test_pred)),
                "balanced_accuracy": float(balanced_accuracy_score(y_test, y_test_pred)),
                "auc": float(roc_auc_score(y_test, y_test_prob)),
                "threshold": float(best_result["threshold"]),
            },
        },
    }

    with args.output_model.open("wb") as file:
        pickle.dump(model_bundle, file)

    print(f"\nModel bundle saved as {args.output_model}")


if __name__ == "__main__":
    main()
