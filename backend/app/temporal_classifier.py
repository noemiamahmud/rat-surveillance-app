"""
PyTorch temporal behavior classifier for rat drug experiment analysis.

Replaces heuristic thresholds with a trained LSTM/Transformer model that
classifies behavior from pose + motion feature sequences.

Supports three modes:
  - "pytorch_temporal": trained LSTM or Transformer model
  - "heuristic": legacy rule-based fallback
  - "simba": SimBA integration (see simba_classifier.py)
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .config import settings


BEHAVIOR_LABELS = [
    "exploration",
    "locomotor_burst",
    "freezing_candidate",
    "grooming_candidate",
    "hesitation",
    "stereotypy_candidate",
    "turning_pattern",
    "resting",
    "monitoring",
]

LABEL_TO_IDX = {label: idx for idx, label in enumerate(BEHAVIOR_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}
NUM_CLASSES = len(BEHAVIOR_LABELS)

# Feature vector: 22 features per frame
FEATURE_NAMES = [
    "speed",
    "motion_score",
    "direction_delta",
    "spatial_span",
    "bbox_area_ratio",
    "pace_confined_score",
    "acceleration",
    "centroid_x_norm",
    "centroid_y_norm",
    "rolling_speed_mean",
    "rolling_speed_std",
    "rolling_direction_mean",
    "rolling_span",
    # Pose features (zero if unavailable)
    "body_curvature",
    "turning_rate",
    "body_length",
    "forepaw_span",
    "hindpaw_span",
    "ear_span",
    "tail_extension",
    "grooming_local_motion",
    "nose_tail_angle",
]
NUM_FEATURES = len(FEATURE_NAMES)

# Default sequence window for temporal context
SEQUENCE_LENGTH = 64


if TORCH_AVAILABLE:

    class BehaviorLSTM(nn.Module):
        """Bidirectional LSTM with attention for frame-level behavior classification."""

        def __init__(
            self,
            input_size: int = NUM_FEATURES,
            hidden_size: int = 128,
            num_layers: int = 2,
            num_classes: int = NUM_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.input_norm = nn.LayerNorm(input_size)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.attention = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.Tanh(),
                nn.Linear(hidden_size, 1),
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Args:
                x: (batch, seq_len, features)
            Returns:
                logits: (batch, seq_len, num_classes)
            """
            x = self.input_norm(x)
            lstm_out, _ = self.lstm(x)  # (batch, seq, hidden*2)
            logits = self.classifier(lstm_out)  # (batch, seq, num_classes)
            return logits

    class BehaviorTransformer(nn.Module):
        """Transformer encoder for temporal behavior classification."""

        def __init__(
            self,
            input_size: int = NUM_FEATURES,
            d_model: int = 128,
            nhead: int = 4,
            num_layers: int = 3,
            num_classes: int = NUM_CLASSES,
            dropout: float = 0.2,
            max_seq_len: int = 2048,
        ):
            super().__init__()
            self.input_proj = nn.Linear(input_size, d_model)
            self.pos_encoding = nn.Parameter(torch.randn(1, max_seq_len, d_model) * 0.02)
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
            )
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
            self.classifier = nn.Sequential(
                nn.LayerNorm(d_model),
                nn.Linear(d_model, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seq_len = x.size(1)
            x = self.input_proj(x)
            x = x + self.pos_encoding[:, :seq_len, :]
            x = self.encoder(x)
            return self.classifier(x)


@dataclass
class FrameClassification:
    label: str
    confidence: float
    logits: list[float]


def extract_features_from_motion(
    *,
    speeds: list[float],
    motion_scores: list[float],
    direction_deltas: list[float],
    spatial_spans: list[float],
    bbox_area_ratios: list[float],
    pace_confined_scores: list[float],
    centroids_x: list[float | None],
    centroids_y: list[float | None],
    frame_width: int,
    frame_height: int,
    fps: float,
    window_frames: int = 30,
) -> np.ndarray:
    """Convert raw motion features to the standardized feature matrix (N, NUM_FEATURES)."""
    n = len(speeds)
    features = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    if n == 0:
        return features

    speeds_arr = np.array(speeds, dtype=np.float32)
    motion_arr = np.array(motion_scores, dtype=np.float32)
    direction_arr = np.array(direction_deltas, dtype=np.float32)
    span_arr = np.array(spatial_spans, dtype=np.float32)
    bbox_arr = np.array(bbox_area_ratios, dtype=np.float32)
    pace_arr = np.array(pace_confined_scores, dtype=np.float32)

    features[:, 0] = speeds_arr
    features[:, 1] = motion_arr
    features[:, 2] = direction_arr
    features[:, 3] = span_arr
    features[:, 4] = bbox_arr
    features[:, 5] = pace_arr

    # Acceleration
    accel = np.diff(speeds_arr, prepend=speeds_arr[0]) * fps
    features[:, 6] = accel

    # Normalized centroids
    cx = np.array([x if x is not None else 0.0 for x in centroids_x], dtype=np.float32)
    cy = np.array([y if y is not None else 0.0 for y in centroids_y], dtype=np.float32)
    features[:, 7] = cx / max(frame_width, 1)
    features[:, 8] = cy / max(frame_height, 1)

    window = max(1, int(window_frames))
    c_speed = np.cumsum(speeds_arr, dtype=np.float64)
    c_speed_sq = np.cumsum(speeds_arr.astype(np.float64) ** 2)
    c_dir = np.cumsum(direction_arr, dtype=np.float64)
    c_span = np.cumsum(span_arr, dtype=np.float64)
    counts = np.arange(1, n + 1, dtype=np.float64)
    start_idx = np.maximum(0, np.arange(n) - window)
    span_counts = counts - start_idx
    prev_c_speed = np.concatenate([[0.0], c_speed[:-1]])
    prev_c_speed_sq = np.concatenate([[0.0], c_speed_sq[:-1]])
    prev_c_dir = np.concatenate([[0.0], c_dir[:-1]])
    prev_c_span = np.concatenate([[0.0], c_span[:-1]])
    start_c_speed = np.where(start_idx > 0, prev_c_speed[start_idx], 0.0)
    start_c_speed_sq = np.where(start_idx > 0, prev_c_speed_sq[start_idx], 0.0)
    start_c_dir = np.where(start_idx > 0, prev_c_dir[start_idx], 0.0)
    start_c_span = np.where(start_idx > 0, prev_c_span[start_idx], 0.0)
    # Inclusive window [start_idx, i]
    sum_speed = c_speed - start_c_speed
    sum_speed_sq = c_speed_sq - start_c_speed_sq
    features[:, 9] = (sum_speed / span_counts).astype(np.float32)
    var = np.maximum(sum_speed_sq / span_counts - features[:, 9] ** 2, 0.0)
    features[:, 10] = np.sqrt(var).astype(np.float32)
    features[:, 11] = ((c_dir - start_c_dir) / span_counts).astype(np.float32)
    features[:, 12] = ((c_span - start_c_span) / span_counts).astype(np.float32)

    # Pose features default to 0 (filled by enrich_with_pose if available)
    return features


def enrich_with_pose(
    features: np.ndarray,
    pose_enrichment: dict | None,
    fps: float,
) -> np.ndarray:
    """Add per-frame pose features when DLC output is available; otherwise leave zeros."""
    if pose_enrichment is None:
        return features

    pose_df = pose_enrichment.get("pose_df")
    if pose_df is not None:
        from .pose_analysis import extract_pose_features_per_frame

        pose_mat = extract_pose_features_per_frame(pose_df, fps=fps, n_frames=len(features))
        features[:, 13:22] = pose_mat
        return features

    summary = pose_enrichment.get("summary", {})
    metrics = summary.get("metrics", {})
    features[:, 13] = metrics.get("mean_body_curvature_deg", 0.0)
    features[:, 14] = metrics.get("mean_turning_rate_deg_s", 0.0)
    features[:, 15] = metrics.get("mean_body_length_px", 0.0)
    features[:, 19] = metrics.get("mean_tail_extension_px", 0.0)
    features[:, 20] = metrics.get("grooming_motion_score", 0.0)

    frame_keypoints = pose_enrichment.get("frame_keypoints", {})
    for frame_idx, points in frame_keypoints.items():
        if frame_idx >= len(features):
            continue
        if "left_forepaw" in points and "right_forepaw" in points:
            lf = points["left_forepaw"]
            rf = points["right_forepaw"]
            features[frame_idx, 16] = np.hypot(lf[0] - rf[0], lf[1] - rf[1])
        if "left_hindpaw" in points and "right_hindpaw" in points:
            lh = points["left_hindpaw"]
            rh = points["right_hindpaw"]
            features[frame_idx, 17] = np.hypot(lh[0] - rh[0], lh[1] - rh[1])
        if "left_ear" in points and "right_ear" in points:
            le = points["left_ear"]
            re = points["right_ear"]
            features[frame_idx, 18] = np.hypot(le[0] - re[0], le[1] - re[1])
        if "nose" in points and "tail_base" in points:
            nose = points["nose"]
            tail = points["tail_base"]
            features[frame_idx, 21] = np.degrees(
                np.arctan2(nose[1] - tail[1], nose[0] - tail[0])
            )

    return features


class TemporalBehaviorClassifier:
    """Wrapper for loading and running the trained temporal classifier."""

    def __init__(
        self,
        model_path: str | None = None,
        model_type: str = "lstm",
        device: str | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for the temporal classifier.")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model_type = model_type

        if model_type == "lstm":
            self.model = BehaviorLSTM()
        else:
            self.model = BehaviorTransformer()

        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict(self, features: np.ndarray) -> list[FrameClassification]:
        """
        Classify behavior for each frame.

        Args:
            features: (num_frames, NUM_FEATURES) feature matrix
        Returns:
            List of FrameClassification per frame
        """
        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            logits = self.model(x)  # (1, seq_len, num_classes)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            logits_np = logits.squeeze(0).cpu().numpy()

        results = []
        for i in range(len(features)):
            pred_idx = int(np.argmax(probs[i]))
            results.append(FrameClassification(
                label=IDX_TO_LABEL[pred_idx],
                confidence=float(probs[i][pred_idx]),
                logits=logits_np[i].tolist(),
            ))
        return results


def heuristic_classify(
    *,
    speed: float,
    motion_score: float,
    direction_delta: float,
    spatial_span: float,
    pace_confined_score: float,
    recent_avg_speed: float,
    recent_direction_mean: float,
    width: int,
) -> tuple[str, float]:
    """Legacy heuristic classification — used as fallback when no trained model exists."""
    if speed > 70 and spatial_span >= 90:
        label = "locomotor_burst"
        confidence = min(0.98, 0.5 + speed / 220.0)
    elif speed > 38 and spatial_span >= 70:
        label = "exploration"
        confidence = min(0.96, 0.45 + spatial_span / max(width, 1))
    elif speed < 5 and motion_score < 1.0 and spatial_span < 26:
        label = "freezing_candidate"
        confidence = 0.84
    elif speed < 9 and motion_score > 2.0 and spatial_span < 22:
        label = "grooming_candidate"
        confidence = min(0.9, 0.48 + motion_score / 15.0)
    elif speed < 15 and recent_avg_speed - speed > 10 and motion_score > 1.5:
        label = "hesitation"
        confidence = min(0.9, 0.44 + (recent_avg_speed - speed) / 55.0)
    elif speed > 20 and spatial_span < 55 and (
        pace_confined_score > 0.9 or recent_direction_mean > 90
    ):
        label = "stereotypy_candidate"
        confidence = min(0.98, 0.52 + pace_confined_score / 3.0)
    elif speed > 18 and recent_direction_mean > 65 and spatial_span < 80:
        label = "turning_pattern"
        confidence = min(0.9, 0.46 + recent_direction_mean / 220.0)
    elif speed < 4 and motion_score < 1.4:
        label = "resting"
        confidence = 0.72
    else:
        label = "monitoring"
        confidence = 0.35
    return label, confidence


def get_classifier(classifier_type: str | None = None) -> TemporalBehaviorClassifier | None:
    """Load the temporal classifier if configured and a trained model exists."""
    ctype = (classifier_type or getattr(settings, "CLASSIFIER_TYPE", "heuristic")).strip().lower()
    if ctype != "pytorch_temporal":
        return None

    if not TORCH_AVAILABLE:
        return None

    model_path = getattr(settings, "CLASSIFIER_MODEL_PATH", None)
    if not model_path or not Path(model_path).exists():
        return None
    model_type = getattr(settings, "CLASSIFIER_MODEL_TYPE", "lstm")
    return TemporalBehaviorClassifier(
        model_path=model_path,
        model_type=model_type,
    )


def save_model_metadata(model_path: str, metadata: dict) -> None:
    """Save training metadata alongside a model checkpoint."""
    meta_path = Path(model_path).with_suffix(".meta.json")
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_model_metadata(model_path: str) -> dict | None:
    """Load training metadata for a model checkpoint."""
    meta_path = Path(model_path).with_suffix(".meta.json")
    if meta_path.exists():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    return None
