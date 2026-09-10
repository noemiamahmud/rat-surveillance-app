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
    BEHAVIOR_LABELS,
    FEATURE_NAMES,
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


HARD_REVIEW_LABELS = {
    "grooming_candidate",
    "freezing_candidate",
    "stereotypy_candidate",
    "turning_pattern",
    "hesitation",
}


def write_training_frames_csv(
    output_path: str | Path,
    *,
    feature_matrix,
    frame_states,
    analysis_id: str,
    condition: str | None,
    fps: float,
    label_source: str,
) -> Path:
    """Write one row per frame: 22 features + label metadata."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = min(len(feature_matrix), len(frame_states))
    rows = []
    for i in range(n):
        state = frame_states[i]
        row = {name: float(feature_matrix[i, idx]) for idx, name in enumerate(FEATURE_NAMES)}
        row.update(
            {
                "timestamp_s": float(state.timestamp_s),
                "label": state.label,
                "confidence": float(state.confidence),
                "analysis_id": analysis_id,
                "condition": condition or "",
                "fps": float(fps),
                "label_source": label_source,
            }
        )
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def collect_session_csvs(sessions_dir: str | Path | None = None) -> list[Path]:
    root = Path(sessions_dir) if sessions_dir else settings_annotated_root()
    return sorted(root.glob("*/training_frames.csv"))


def settings_annotated_root() -> Path:
    from .config import settings

    return Path(settings.ANNOTATED_OUTPUTS_DIR)


def export_dataset(
    output_path: str | Path,
    sessions_dir: str | Path | None = None,
) -> dict:
    """Concatenate per-session training_frames.csv files into one dataset."""
    csvs = collect_session_csvs(sessions_dir)
    if not csvs:
        raise FileNotFoundError(
            "No training_frames.csv files found. Analyze videos in the app first "
            "(each run writes backend/data/processed/annotated_sessions/<id>/training_frames.csv)."
        )
    frames = [pd.read_csv(path) for path in csvs]
    dataset = pd.concat(frames, ignore_index=True)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(path, index=False)
    counts = dataset["label"].value_counts().to_dict()
    return {
        "output_path": str(path),
        "sessions": len(csvs),
        "frames": int(len(dataset)),
        "label_counts": {str(k): int(v) for k, v in counts.items()},
    }


def export_review_queue(
    output_path: str | Path,
    sessions_dir: str | Path | None = None,
    *,
    min_confidence: float = 0.6,
    min_bout_s: float = 0.4,
) -> dict:
    """
    Bout-level review CSV. Fill `corrected_label` only where the HMM is wrong.
    Leave it blank to keep the auto label.
    """
    root = Path(sessions_dir) if sessions_dir else settings_annotated_root()
    rows = []
    for timeline_path in sorted(root.glob("*/timeline.json")):
        data = json.loads(timeline_path.read_text(encoding="utf-8"))
        analysis_id = data.get("analysis_id") or timeline_path.parent.name
        condition = data.get("condition") or ""
        for bout in data.get("bouts", []):
            label = str(bout.get("label", "monitoring"))
            confidence = float(bout.get("confidence", 0.0))
            duration = float(bout.get("duration_s", 0.0))
            hard = label in HARD_REVIEW_LABELS or confidence < min_confidence
            if not hard or duration < min_bout_s:
                continue
            rows.append(
                {
                    "analysis_id": analysis_id,
                    "condition": condition,
                    "start_s": round(float(bout["start_s"]), 3),
                    "end_s": round(float(bout["end_s"]), 3),
                    "duration_s": round(duration, 3),
                    "predicted_label": label,
                    "confidence": round(confidence, 3),
                    "corrected_label": "",
                    "notes": "",
                }
            )
    if not rows:
        raise ValueError("No review bouts found. Analyze at least one video first.")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return {"output_path": str(path), "bouts": len(rows)}


def apply_review(
    review_csv: str | Path,
    sessions_dir: str | Path | None = None,
) -> dict:
    """Overwrite frame labels in training_frames.csv using corrected_label rows."""
    review = pd.read_csv(review_csv)
    if "corrected_label" not in review.columns:
        raise ValueError("review CSV must have a corrected_label column.")
    root = Path(sessions_dir) if sessions_dir else settings_annotated_root()
    valid = {label.lower() for label in BEHAVIOR_LABELS}
    patched_frames = 0
    session_writes: set[str] = set()
    review = review.copy()
    review["corrected_label"] = review["corrected_label"].fillna("").astype(str).str.strip()
    edited = review[~review["corrected_label"].str.lower().isin({"", "nan", "none"})]
    for _, row in edited.iterrows():
        label = str(row["corrected_label"]).strip().lower()
        if label not in valid:
            raise ValueError(f"Unknown corrected_label '{label}'. Allowed: {sorted(valid)}")
        session_csv = root / str(row["analysis_id"]) / "training_frames.csv"
        if not session_csv.exists():
            continue
        frames = pd.read_csv(session_csv)
        mask = (frames["timestamp_s"] >= float(row["start_s"])) & (
            frames["timestamp_s"] <= float(row["end_s"])
        )
        n = int(mask.sum())
        if n == 0:
            continue
        frames.loc[mask, "label"] = label
        frames.loc[mask, "label_source"] = "human"
        frames.to_csv(session_csv, index=False)
        session_writes.add(str(row["analysis_id"]))
        patched_frames += n
    return {
        "corrected_bouts": int(len(edited)),
        "patched_session_writes": len(session_writes),
        "patched_frames": patched_frames,
    }


def load_labeled_dataset(csv_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a labeled behavior dataset CSV.

    Expected format: FEATURE_NAMES columns + a 'label' column.
    Extra metadata columns are ignored.
    """
    df = pd.read_csv(csv_path)

    if "label" not in df.columns:
        raise ValueError(f"CSV must have a 'label' column. Found: {list(df.columns)}")

    missing = [name for name in FEATURE_NAMES if name not in df.columns]
    if missing:
        feature_cols = [col for col in df.columns if col not in {"label", "timestamp_s", "confidence", "analysis_id", "condition", "fps", "label_source"}]
        features = df[feature_cols].values.astype(np.float32)
        if features.shape[1] < NUM_FEATURES:
            padding = np.zeros((features.shape[0], NUM_FEATURES - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=1)
        elif features.shape[1] > NUM_FEATURES:
            features = features[:, :NUM_FEATURES]
    else:
        features = df[FEATURE_NAMES].values.astype(np.float32)

    labels = np.array(
        [
            LABEL_TO_IDX.get(str(label).strip().lower(), LABEL_TO_IDX["monitoring"])
            for label in df["label"]
        ],
        dtype=np.int64,
    )

    return features, labels


def split_train_val(
    csv_path: str,
    val_fraction: float = 0.2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split by analysis_id when present, else a contiguous time split (no shuffled frames)."""
    df = pd.read_csv(csv_path)
    if "analysis_id" in df.columns and df["analysis_id"].nunique() >= 2:
        sessions = list(df["analysis_id"].unique())
        rng = np.random.default_rng(42)
        rng.shuffle(sessions)
        n_val = max(1, int(round(len(sessions) * val_fraction)))
        val_ids = set(sessions[:n_val])
        train_df = df[~df["analysis_id"].isin(val_ids)]
        val_df = df[df["analysis_id"].isin(val_ids)]
        if train_df.empty or val_df.empty:
            train_df, val_df = _contiguous_split(df, val_fraction)
    else:
        train_df, val_df = _contiguous_split(df, val_fraction)

    return (*_dataframe_to_xy(train_df), *_dataframe_to_xy(val_df))


def _contiguous_split(df: pd.DataFrame, val_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    cut = max(1, min(len(df) - 1, int(len(df) * (1.0 - val_fraction))))
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def _dataframe_to_xy(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if df.empty:
        return np.zeros((0, NUM_FEATURES), dtype=np.float32), np.zeros((0,), dtype=np.int64)
    if all(name in df.columns for name in FEATURE_NAMES):
        features = df[FEATURE_NAMES].values.astype(np.float32)
    else:
        feature_cols = [
            col
            for col in df.columns
            if col not in {"label", "timestamp_s", "confidence", "analysis_id", "condition", "fps", "label_source"}
        ]
        features = df[feature_cols].values.astype(np.float32)
        if features.shape[1] < NUM_FEATURES:
            padding = np.zeros((features.shape[0], NUM_FEATURES - features.shape[1]), dtype=np.float32)
            features = np.concatenate([features, padding], axis=1)
        else:
            features = features[:, :NUM_FEATURES]
    labels = np.array(
        [
            LABEL_TO_IDX.get(str(label).strip().lower(), LABEL_TO_IDX["monitoring"])
            for label in df["label"]
        ],
        dtype=np.int64,
    )
    return features, labels


def train_from_csv(
    csv_path: str,
    *,
    model_type: str = "lstm",
    epochs: int = 40,
    batch_size: int = 16,
    learning_rate: float = 1e-3,
    output_dir: str | None = None,
    val_fraction: float = 0.2,
) -> dict:
    train_features, train_labels, val_features, val_labels = split_train_val(
        csv_path, val_fraction=val_fraction
    )
    if len(train_features) < 8:
        raise ValueError(
            f"Not enough training frames ({len(train_features)}). Analyze more videos first."
        )
    return train_model(
        train_features=train_features,
        train_labels=train_labels,
        val_features=val_features if len(val_features) else None,
        val_labels=val_labels if len(val_labels) else None,
        model_type=model_type,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        output_dir=output_dir,
    )


def generate_weak_labels_from_heuristic(
    motion_data_path: str,
    output_path: str,
) -> str:
    """
    Backward-compatible export from timeline.json.

    Prefer training_frames.csv from a new analysis run — it includes all 22 features.
    """
    timeline_path = Path(motion_data_path)
    session_csv = timeline_path.parent / "training_frames.csv"
    if session_csv.exists():
        df = pd.read_csv(session_csv)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        return output_path

    data = json.loads(timeline_path.read_text(encoding="utf-8"))
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
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return output_path


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

    from .temporal_classifier import BehaviorLSTM, BehaviorTransformer

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

    # Normalize
    train_features, scaler = normalize_features(train_features)
    if val_features is not None:
        val_features, _ = normalize_features(val_features, scaler)

    # Datasets
    train_dataset = BehaviorSequenceDataset(train_features, train_labels)
    drop_last = len(train_dataset) > batch_size
    train_loader = DataLoader(
        train_dataset,
        batch_size=min(batch_size, max(len(train_dataset), 1)),
        shuffle=True,
        drop_last=drop_last,
    )

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

        print(
            f"epoch {epoch + 1}/{epochs}  train_loss={avg_train_loss:.4f}  "
            f"val_loss={val_loss if val_loss != float('inf') else float('nan'):.4f}  "
            f"val_acc={val_acc:.3f}"
        )

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
