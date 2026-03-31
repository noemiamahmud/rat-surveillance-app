"""
Training pipeline for the temporal behavior classifier.

Handles:
  - Loading labeled behavior datasets (CSV format)
  - Feature extraction and normalization
  - Train/val splitting with stratification
  - LSTM and Transformer training loops
  - Model checkpoint saving with metadata
  - Cross-validation support
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .temporal_classifier import (
    BehaviorLSTM,
    BehaviorTransformer,
    BEHAVIOR_LABELS,
    LABEL_TO_IDX,
    NUM_FEATURES,
    NUM_CLASSES,
    SEQUENCE_LENGTH,
    save_model_metadata,
)


LABELED_DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "labeled"


if TORCH_AVAILABLE:

    class BehaviorSequenceDataset(Dataset):
        """Dataset of fixed-length feature sequences with frame-level labels."""

        def __init__(
            self,
            features: np.ndarray,
            labels: np.ndarray,
            sequence_length: int = SEQUENCE_LENGTH,
            stride: int = 16,
        ):
            self.features = features.astype(np.float32)
            self.labels = labels.astype(np.int64)
            self.sequence_length = sequence_length
            self.stride = stride
            self.indices = []

            n = len(features)
            for start in range(0, max(1, n - sequence_length + 1), stride):
                end = min(start + sequence_length, n)
                self.indices.append((start, end))

        def __len__(self) -> int:
            return len(self.indices)

        def __getitem__(self, idx: int):
            start, end = self.indices[idx]
            seq_features = self.features[start:end]
            seq_labels = self.labels[start:end]

            # Pad if shorter than sequence_length
            if len(seq_features) < self.sequence_length:
                pad_len = self.sequence_length - len(seq_features)
                seq_features = np.pad(seq_features, ((0, pad_len), (0, 0)), mode="edge")
                seq_labels = np.pad(seq_labels, (0, pad_len), mode="edge")

            return torch.from_numpy(seq_features), torch.from_numpy(seq_labels)


def load_labeled_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a labeled behavior dataset CSV.

    Expected format: columns matching FEATURE_NAMES + a 'label' column.
    Returns (features, labels) as numpy arrays.
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(f"CSV must have a 'label' column. Found: {list(df.columns)}")

    feature_cols = [col for col in df.columns if col != "label" and col != "timestamp_s"]
    features = df[feature_cols].values.astype(np.float32)

    # Pad or truncate features to NUM_FEATURES
    if features.shape[1] < NUM_FEATURES:
        padding = np.zeros((features.shape[0], NUM_FEATURES - features.shape[1]), dtype=np.float32)
        features = np.concatenate([features, padding], axis=1)
    elif features.shape[1] > NUM_FEATURES:
        features = features[:, :NUM_FEATURES]

    labels = np.array([
        LABEL_TO_IDX.get(label.strip().lower(), LABEL_TO_IDX["monitoring"])
        for label in df["label"]
    ], dtype=np.int64)

    return features, labels


def compute_class_weights(labels: np.ndarray) -> torch.Tensor:
    """Compute inverse-frequency class weights for imbalanced behavior labels."""
    counts = np.bincount(labels, minlength=NUM_CLASSES).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * NUM_CLASSES
    return torch.from_numpy(weights)


def normalize_features(
    features: np.ndarray,
    scaler: Optional[object] = None,
) -> tuple[np.ndarray, object]:
    """Standardize features. Returns normalized features and fitted scaler."""
    if not SKLEARN_AVAILABLE:
        mean = features.mean(axis=0)
        std = features.std(axis=0)
        std[std < 1e-8] = 1.0
        return (features - mean) / std, {"mean": mean, "std": std}

    if scaler is None:
        scaler = StandardScaler()
        features = scaler.fit_transform(features)
    else:
        features = scaler.transform(features)
    return features, scaler


def train_model(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    val_features: np.ndarray | None = None,
    val_labels: np.ndarray | None = None,
    model_type: str = "lstm",
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    device: str | None = None,
    output_dir: str | None = None,
    condition: str | None = None,
) -> dict:
    """
    Train the temporal behavior classifier.

    Returns dict with training history and best model path.
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for training.")

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Normalize
    train_features, scaler = normalize_features(train_features)
    if val_features is not None:
        val_features, _ = normalize_features(val_features, scaler)

    # Datasets
    train_dataset = BehaviorSequenceDataset(train_features, train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    val_loader = None
    if val_features is not None and val_labels is not None:
        val_dataset = BehaviorSequenceDataset(val_features, val_labels, stride=SEQUENCE_LENGTH)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Model
    if model_type == "transformer":
        model = BehaviorTransformer()
    else:
        model = BehaviorLSTM()
    model = model.to(device)

    # Loss with class weighting
    class_weights = compute_class_weights(train_labels).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Training loop
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    best_val_loss = float("inf")
    best_state = None

    out_dir = Path(output_dir or LABELED_DATA_DIR.parent / "models" / "behavior_classifier")
    out_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        batches = 0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            optimizer.zero_grad()
            logits = model(batch_features)  # (batch, seq, classes)
            loss = criterion(logits.reshape(-1, NUM_CLASSES), batch_labels.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            batches += 1

        scheduler.step()
        avg_train_loss = epoch_loss / max(batches, 1)
        history["train_loss"].append(avg_train_loss)

        # Validation
        val_loss = float("inf")
        val_acc = 0.0
        if val_loader is not None:
            model.eval()
            val_total_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch_features, batch_labels in val_loader:
                    batch_features = batch_features.to(device)
                    batch_labels = batch_labels.to(device)
                    logits = model(batch_features)
                    loss = criterion(logits.reshape(-1, NUM_CLASSES), batch_labels.reshape(-1))
                    val_total_loss += loss.item()
                    preds = logits.argmax(dim=-1)
                    val_correct += (preds == batch_labels).sum().item()
                    val_total += batch_labels.numel()

            val_loss = val_total_loss / max(len(val_loader), 1)
            val_acc = val_correct / max(val_total, 1)

        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Save best model
    model_path = out_dir / f"behavior_{model_type}_best.pt"
    if best_state is not None:
        torch.save(best_state, model_path)
    else:
        torch.save(model.state_dict(), model_path)

    # Save metadata
    metadata = {
        "model_type": model_type,
        "num_features": NUM_FEATURES,
        "num_classes": NUM_CLASSES,
        "labels": BEHAVIOR_LABELS,
        "epochs_trained": epochs,
        "best_val_loss": best_val_loss,
        "best_val_accuracy": history["val_accuracy"][-1] if history["val_accuracy"] else None,
        "condition": condition,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    save_model_metadata(str(model_path), metadata)

    # Save scaler
    if SKLEARN_AVAILABLE and hasattr(scaler, "mean_"):
        import pickle
        scaler_path = out_dir / "feature_scaler.pkl"
        with open(scaler_path, "wb") as f:
            pickle.dump(scaler, f)

    return {
        "model_path": str(model_path),
        "history": history,
        "metadata": metadata,
    }


def cross_validate(
    features: np.ndarray,
    labels: np.ndarray,
    n_folds: int = 5,
    model_type: str = "lstm",
    epochs: int = 30,
    device: str | None = None,
) -> dict:
    """Run stratified k-fold cross-validation."""
    if not SKLEARN_AVAILABLE:
        raise RuntimeError("scikit-learn is required for cross-validation.")
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch is required for cross-validation.")

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(features, labels)):
        result = train_model(
            train_features=features[train_idx],
            train_labels=labels[train_idx],
            val_features=features[val_idx],
            val_labels=labels[val_idx],
            model_type=model_type,
            epochs=epochs,
            device=device,
        )
        fold_results.append({
            "fold": fold,
            "best_val_loss": result["metadata"]["best_val_loss"],
            "best_val_accuracy": result["metadata"]["best_val_accuracy"],
        })

    return {
        "n_folds": n_folds,
        "folds": fold_results,
        "mean_val_accuracy": float(np.mean([f["best_val_accuracy"] or 0 for f in fold_results])),
        "std_val_accuracy": float(np.std([f["best_val_accuracy"] or 0 for f in fold_results])),
    }


def generate_weak_labels_from_heuristic(
    motion_data_path: str,
    output_path: str,
) -> str:
    """
    Generate weak training labels from heuristic classifier output.

    Takes a timeline.json from the heuristic pipeline and converts it to
    a labeled CSV suitable for training.
    """
    data = json.loads(Path(motion_data_path).read_text(encoding="utf-8"))
    frames = data.get("frames", [])
    if not frames:
        raise ValueError("No frame data in timeline.")

    rows = []
    for frame in frames:
        rows.append({
            "speed": frame.get("speed", 0.0),
            "motion_score": frame.get("motion_score", 0.0),
            "label": frame.get("label", "monitoring"),
        })

    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    return output_path
