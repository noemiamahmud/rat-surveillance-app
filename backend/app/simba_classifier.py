"""
SimBA (Simple Behavioral Analysis) integration for rat behavior classification.

SimBA provides LSTM-based supervised behavior classification from DeepLabCut
pose estimation outputs. This module wraps SimBA's workflow for use in the
rat behavior analysis pipeline.

Pipeline: Video → DLC pose extraction → SimBA feature engineering → SimBA classifier → predictions

SimBA reference: https://github.com/sgoldenlab/simba
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import settings
from .pose_analysis import (
    flatten_dlc_columns,
    compute_centroid_track,
    infer_bodyparts,
    available_skeleton_pairs,
    LIKELIHOOD_THRESHOLD,
)
from .temporal_classifier import (
    BEHAVIOR_LABELS,
    LABEL_TO_IDX,
    IDX_TO_LABEL,
    NUM_CLASSES,
)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# SimBA-style features computed from pose sequences
SIMBA_FEATURE_NAMES = [
    # Distance features (pairwise distances between all bodypart combinations)
    "nose_to_tail_base_dist",
    "nose_to_left_ear_dist",
    "nose_to_right_ear_dist",
    "left_ear_to_right_ear_dist",
    "left_forepaw_to_right_forepaw_dist",
    "left_hindpaw_to_right_hindpaw_dist",
    "shoulder_to_hip_dist",
    "nose_to_hip_dist",
    # Velocity features
    "nose_velocity",
    "tail_base_velocity",
    "centroid_velocity",
    "body_length_change_rate",
    # Angular features
    "body_axis_angle",
    "body_axis_angular_velocity",
    "head_direction_angle",
    "tail_curvature_angle",
    # Area features
    "body_area_convex_hull",
    "body_elongation_ratio",
    # Rolling statistics (2-second windows)
    "centroid_velocity_mean_2s",
    "centroid_velocity_std_2s",
    "body_curvature_mean_2s",
    "angular_velocity_mean_2s",
    "distance_traveled_2s",
    "spatial_span_2s",
]
NUM_SIMBA_FEATURES = len(SIMBA_FEATURE_NAMES)


def _safe_distance(df: pd.DataFrame, part_a: str, part_b: str) -> pd.Series:
    """Compute Euclidean distance between two bodyparts, returning NaN where data is missing."""
    ax_col, ay_col = f"{part_a}_x", f"{part_a}_y"
    bx_col, by_col = f"{part_b}_x", f"{part_b}_y"
    if ax_col not in df or bx_col not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    al_col = f"{part_a}_likelihood"
    bl_col = f"{part_b}_likelihood"
    mask = pd.Series(True, index=df.index)
    if al_col in df:
        mask &= df[al_col].fillna(0) >= LIKELIHOOD_THRESHOLD
    if bl_col in df:
        mask &= df[bl_col].fillna(0) >= LIKELIHOOD_THRESHOLD
    dist = np.sqrt((df[ax_col] - df[bx_col]) ** 2 + (df[ay_col] - df[by_col]) ** 2)
    return dist.where(mask)


def _safe_velocity(df: pd.DataFrame, part: str, dt: float) -> pd.Series:
    """Compute velocity of a bodypart in pixels/second."""
    x_col, y_col = f"{part}_x", f"{part}_y"
    if x_col not in df or y_col not in df:
        return pd.Series(0.0, index=df.index, dtype=float)
    dx = df[x_col].diff().fillna(0)
    dy = df[y_col].diff().fillna(0)
    return np.sqrt(dx ** 2 + dy ** 2) / max(dt, 1e-6)


def extract_simba_features(
    df: pd.DataFrame,
    fps: float = 30.0,
) -> np.ndarray:
    """
    Extract SimBA-style features from a DLC pose DataFrame.

    Returns (n_frames, NUM_SIMBA_FEATURES) feature matrix.
    """
    flattened = flatten_dlc_columns(df)
    n = len(flattened)
    dt = 1.0 / fps if fps > 0 else 1.0 / 30.0
    rolling_window = max(5, int(2.0 / dt))

    features = np.zeros((n, NUM_SIMBA_FEATURES), dtype=np.float32)

    # Distance features
    features[:, 0] = _safe_distance(flattened, "nose", "tail_base").fillna(0).values
    features[:, 1] = _safe_distance(flattened, "nose", "left_ear").fillna(0).values
    features[:, 2] = _safe_distance(flattened, "nose", "right_ear").fillna(0).values
    features[:, 3] = _safe_distance(flattened, "left_ear", "right_ear").fillna(0).values
    features[:, 4] = _safe_distance(flattened, "left_forepaw", "right_forepaw").fillna(0).values
    features[:, 5] = _safe_distance(flattened, "left_hindpaw", "right_hindpaw").fillna(0).values
    features[:, 6] = _safe_distance(flattened, "shoulder", "hip").fillna(0).values
    features[:, 7] = _safe_distance(flattened, "nose", "hip").fillna(0).values

    # Velocity features
    features[:, 8] = _safe_velocity(flattened, "nose", dt).fillna(0).values
    features[:, 9] = _safe_velocity(flattened, "tail_base", dt).fillna(0).values

    try:
        centroid, _ = compute_centroid_track(flattened)
        cx_diff = centroid["x_smooth"].diff().fillna(0)
        cy_diff = centroid["y_smooth"].diff().fillna(0)
        centroid_vel = np.sqrt(cx_diff ** 2 + cy_diff ** 2) / dt
        features[:, 10] = centroid_vel.values

        # Rolling centroid stats
        features[:, 18] = centroid_vel.rolling(window=rolling_window, min_periods=1).mean().values
        features[:, 19] = centroid_vel.rolling(window=rolling_window, min_periods=1).std().fillna(0).values
        displacement = np.sqrt(cx_diff ** 2 + cy_diff ** 2)
        features[:, 22] = displacement.rolling(window=rolling_window, min_periods=1).sum().values

        x_span = centroid["x_smooth"].rolling(window=rolling_window, min_periods=1).apply(
            lambda s: s.max() - s.min(), raw=True
        ).fillna(0)
        y_span = centroid["y_smooth"].rolling(window=rolling_window, min_periods=1).apply(
            lambda s: s.max() - s.min(), raw=True
        ).fillna(0)
        features[:, 23] = np.sqrt(x_span.values ** 2 + y_span.values ** 2)
    except (ValueError, KeyError):
        pass

    # Body length change rate
    body_length = _safe_distance(flattened, "nose", "tail_base").fillna(method="ffill").fillna(0)
    features[:, 11] = (body_length.diff().fillna(0) / max(dt, 1e-6)).values

    # Angular features
    for nose_col, tail_col in [("nose", "tail_base")]:
        nx, ny = f"{nose_col}_x", f"{nose_col}_y"
        tx, ty = f"{tail_col}_x", f"{tail_col}_y"
        if all(c in flattened for c in [nx, ny, tx, ty]):
            angle = np.degrees(np.arctan2(
                flattened[ny] - flattened[ty],
                flattened[nx] - flattened[tx],
            ))
            features[:, 12] = angle.fillna(0).values
            unwrapped = np.unwrap(np.radians(angle.fillna(0).values))
            angular_vel = np.abs(np.diff(unwrapped, prepend=unwrapped[0])) / max(dt, 1e-6)
            features[:, 13] = np.degrees(angular_vel)

    # Head direction
    for left, right in [("left_ear", "right_ear")]:
        lx, ly = f"{left}_x", f"{left}_y"
        rx, ry = f"{right}_x", f"{right}_y"
        nx_col, ny_col = "nose_x", "nose_y"
        if all(c in flattened for c in [lx, ly, rx, ry, nx_col, ny_col]):
            mid_x = (flattened[lx] + flattened[rx]) / 2
            mid_y = (flattened[ly] + flattened[ry]) / 2
            head_angle = np.degrees(np.arctan2(
                flattened[ny_col] - mid_y,
                flattened[nx_col] - mid_x,
            ))
            features[:, 14] = head_angle.fillna(0).values

    # Tail curvature
    tb_x, tb_y = "tail_base_x", "tail_base_y"
    tm_x, tm_y = "tail_mid_x", "tail_mid_y"
    tt_x, tt_y = "tail_tip_x", "tail_tip_y"
    if all(c in flattened for c in [tb_x, tb_y, tm_x, tm_y, tt_x, tt_y]):
        a1 = np.arctan2(flattened[tm_y] - flattened[tb_y], flattened[tm_x] - flattened[tb_x])
        a2 = np.arctan2(flattened[tt_y] - flattened[tm_y], flattened[tt_x] - flattened[tm_x])
        tail_curve = np.abs(np.degrees(np.unwrap((a2 - a1).fillna(0).values)))
        features[:, 15] = tail_curve

    # Body area (convex hull approximation using bodypart extremes)
    bodyparts = infer_bodyparts(flattened.columns)
    all_x = []
    all_y = []
    for bp in bodyparts:
        if f"{bp}_x" in flattened and f"{bp}_y" in flattened:
            all_x.append(flattened[f"{bp}_x"])
            all_y.append(flattened[f"{bp}_y"])
    if all_x:
        x_stack = pd.concat(all_x, axis=1)
        y_stack = pd.concat(all_y, axis=1)
        x_range = x_stack.max(axis=1) - x_stack.min(axis=1)
        y_range = y_stack.max(axis=1) - y_stack.min(axis=1)
        features[:, 16] = (x_range * y_range).fillna(0).values  # bounding box area
        features[:, 17] = (x_range / y_range.replace(0, np.nan)).fillna(1.0).values  # elongation

    # Rolling body curvature and angular velocity
    features[:, 20] = pd.Series(features[:, 15]).rolling(window=rolling_window, min_periods=1).mean().fillna(0).values
    features[:, 21] = pd.Series(features[:, 13]).rolling(window=rolling_window, min_periods=1).mean().fillna(0).values

    # Replace NaN/inf
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


if TORCH_AVAILABLE:

    class SimBAClassifier(nn.Module):
        """
        LSTM-based behavioral classifier inspired by SimBA's architecture.

        Takes pose-derived feature sequences and outputs frame-level behavior labels.
        """

        def __init__(
            self,
            input_size: int = NUM_SIMBA_FEATURES,
            hidden_size: int = 256,
            num_layers: int = 2,
            num_classes: int = NUM_CLASSES,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.feature_norm = nn.BatchNorm1d(input_size)
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.ReLU(),
                nn.Dropout(dropout * 0.5),
                nn.Linear(hidden_size // 2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            # x: (batch, seq_len, features)
            batch_size, seq_len, feat_size = x.shape
            x_flat = x.reshape(-1, feat_size)
            x_normed = self.feature_norm(x_flat)
            x = x_normed.reshape(batch_size, seq_len, feat_size)
            lstm_out, _ = self.lstm(x)
            return self.classifier(lstm_out)


class SimBABehaviorClassifier:
    """
    High-level SimBA classifier wrapper for the analysis pipeline.

    Loads a trained SimBA model and classifies behavior from DLC pose CSVs.
    """

    def __init__(
        self,
        model_path: str | None = None,
        device: str | None = None,
    ):
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for SimBA classifier.")

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.model = SimBAClassifier()
        if model_path and Path(model_path).exists():
            state_dict = torch.load(model_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

    def predict_from_dlc_csv(
        self,
        csv_path: str,
        fps: float = 30.0,
    ) -> list[dict]:
        """
        Classify behavior from a DLC CSV file.

        Returns list of frame-level predictions with label, confidence, and logits.
        """
        from .pose_analysis import load_dlc_csv
        df = load_dlc_csv(csv_path)
        return self.predict_from_dataframe(df, fps=fps)

    def predict_from_dataframe(
        self,
        df: pd.DataFrame,
        fps: float = 30.0,
    ) -> list[dict]:
        """Classify behavior from a DLC DataFrame."""
        features = extract_simba_features(df, fps=fps)

        with torch.no_grad():
            x = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            logits = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()
            logits_np = logits.squeeze(0).cpu().numpy()

        results = []
        for i in range(len(features)):
            pred_idx = int(np.argmax(probs[i]))
            results.append({
                "frame": i,
                "timestamp_s": i / fps,
                "label": IDX_TO_LABEL[pred_idx],
                "confidence": float(probs[i][pred_idx]),
                "logits": logits_np[i].tolist(),
            })
        return results

    def predict_with_quadrant_analysis(
        self,
        df: pd.DataFrame,
        fps: float = 30.0,
        quadrant_regions: dict | None = None,
    ) -> dict:
        """
        Run behavior classification plus quadrant time analysis.

        Quadrant time analysis measures time spent in each region of interest,
        comparable to Med-PC lever response data from operant chambers.

        Args:
            df: DLC pose DataFrame
            fps: video frame rate
            quadrant_regions: dict mapping region names to (x_min, y_min, x_max, y_max)
        """
        predictions = self.predict_from_dataframe(df, fps=fps)

        # Compute centroid for quadrant assignment
        flattened = flatten_dlc_columns(df)
        try:
            centroid, bodyparts = compute_centroid_track(flattened)
        except ValueError:
            return {"predictions": predictions, "quadrant_time": {}}

        # Quadrant time analysis
        quadrant_time: dict[str, float] = {}
        if quadrant_regions:
            dt = 1.0 / fps
            for region_name, (x_min, y_min, x_max, y_max) in quadrant_regions.items():
                in_region = (
                    (centroid["x_smooth"] >= x_min)
                    & (centroid["x_smooth"] <= x_max)
                    & (centroid["y_smooth"] >= y_min)
                    & (centroid["y_smooth"] <= y_max)
                )
                quadrant_time[region_name] = float(in_region.sum() * dt)

        # Behavior distribution
        from collections import Counter
        label_counts = Counter(p["label"] for p in predictions)
        total = len(predictions)
        behavior_share = {
            label: count / max(total, 1)
            for label, count in label_counts.items()
        }

        return {
            "predictions": predictions,
            "quadrant_time": quadrant_time,
            "behavior_share": behavior_share,
            "total_frames": total,
            "duration_s": total / fps,
        }


def get_simba_classifier(
    model_path: str | None = None,
    classifier_type: str | None = None,
) -> SimBABehaviorClassifier | None:
    """Load SimBA classifier if a trained checkpoint is configured."""
    ctype = (classifier_type or getattr(settings, "CLASSIFIER_TYPE", "heuristic")).strip().lower()
    if ctype != "simba":
        return None

    if not TORCH_AVAILABLE:
        return None

    path = model_path or getattr(settings, "SIMBA_MODEL_PATH", None)
    if not path or not Path(path).exists():
        return None
    return SimBABehaviorClassifier(model_path=path)
