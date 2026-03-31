from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


LIKELIHOOD_THRESHOLD = 0.6
MOVEMENT_SPEED_THRESHOLD = 15.0
IMMOBILITY_SPEED_THRESHOLD = 5.0
MIN_BOUT_DURATION_S = 1.0

RAT_BODYPOINT_PRIORITY = [
    "nose",
    "left_ear",
    "right_ear",
    "neck",
    "shoulder",
    "upper_back",
    "spine_mid",
    "lower_back",
    "hip",
    "tail_base",
    "tail_mid",
    "tail_tip",
    "left_forepaw",
    "right_forepaw",
    "left_hindpaw",
    "right_hindpaw",
]

BODYPOINT_ALIASES = {
    "tailbase": "tail_base",
    "tailmid": "tail_mid",
    "tailtip": "tail_tip",
    "upperback": "upper_back",
    "lowerback": "lower_back",
    "spinemid": "spine_mid",
    "leftforepaw": "left_forepaw",
    "rightforepaw": "right_forepaw",
    "lefthindpaw": "left_hindpaw",
    "righthindpaw": "right_hindpaw",
    "leftear": "left_ear",
    "rightear": "right_ear",
}

RAT_SKELETON_CANDIDATES = [
    ("nose", "left_ear"),
    ("nose", "right_ear"),
    ("nose", "neck"),
    ("nose", "upper_back"),
    ("left_ear", "right_ear"),
    ("neck", "shoulder"),
    ("shoulder", "upper_back"),
    ("upper_back", "spine_mid"),
    ("upper_back", "lower_back"),
    ("spine_mid", "lower_back"),
    ("lower_back", "hip"),
    ("hip", "tail_base"),
    ("lower_back", "tail_base"),
    ("tail_base", "tail_mid"),
    ("tail_mid", "tail_tip"),
    ("shoulder", "left_forepaw"),
    ("shoulder", "right_forepaw"),
    ("hip", "left_hindpaw"),
    ("hip", "right_hindpaw"),
]

POSE_BEHAVIOR_PRIORITY = [
    "stereotypy_candidate",
    "freezing_candidate",
    "grooming_candidate",
    "turning_pattern",
    "exploration",
    "locomotor_burst",
    "hesitation",
    "resting",
]


@dataclass
class BoutRecord:
    label: str
    start_s: float
    end_s: float
    duration_s: float
    mean_speed_px_s: float
    confidence: float


def canonicalize_bodypart_name(name: str) -> str:
    normalized = name.strip().lower().replace(" ", "_")
    return BODYPOINT_ALIASES.get(normalized, normalized)


def flatten_dlc_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        flattened_df = df.copy()
        flattened_df.columns = [canonicalize_bodypart_column(str(column)) for column in df.columns]
        return flattened_df

    flattened = []
    for column in df.columns:
        if len(column) == 3:
            _, bodypart, coord = column
            flattened.append(canonicalize_bodypart_column(f"{bodypart}_{coord}"))
        else:
            flattened.append(canonicalize_bodypart_column("_".join(str(value) for value in column if value)))

    result = df.copy()
    result.columns = flattened
    return result


def canonicalize_bodypart_column(column: str) -> str:
    for suffix in ("_x", "_y", "_likelihood"):
        if column.endswith(suffix):
            return f"{canonicalize_bodypart_name(column[:-len(suffix)])}{suffix}"
    return column


def load_dlc_csv(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"DeepLabCut CSV not found: {path}")

    df = pd.read_csv(path, header=[0, 1, 2])
    return flatten_dlc_columns(df)


def infer_bodyparts(columns: Iterable[str]) -> list[str]:
    bodyparts = set()
    for column in columns:
        if column.endswith("_x") or column.endswith("_y") or column.endswith("_likelihood"):
            bodyparts.add(canonicalize_bodypart_name(column.rsplit("_", 1)[0]))

    ordered = [name for name in RAT_BODYPOINT_PRIORITY if name in bodyparts]
    extras = sorted(bodyparts - set(ordered))
    return ordered + extras


def available_skeleton_pairs(bodyparts: Iterable[str]) -> list[tuple[str, str]]:
    bodypart_set = set(bodyparts)
    return [
        (start, end)
        for start, end in RAT_SKELETON_CANDIDATES
        if start in bodypart_set and end in bodypart_set
    ]


def _valid_bodypart_mask(df: pd.DataFrame, bodypart: str) -> pd.Series:
    likelihood_column = f"{bodypart}_likelihood"
    x_column = f"{bodypart}_x"
    y_column = f"{bodypart}_y"

    if x_column not in df or y_column not in df:
        return pd.Series(False, index=df.index)

    mask = df[x_column].notna() & df[y_column].notna()
    if likelihood_column in df:
        mask &= df[likelihood_column].fillna(0) >= LIKELIHOOD_THRESHOLD
    return mask


def compute_centroid_track(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    bodyparts = infer_bodyparts(df.columns)
    if not bodyparts:
        raise ValueError("No DeepLabCut bodypart columns were found in the CSV.")

    x_series = []
    y_series = []
    used_bodyparts = []
    for bodypart in bodyparts:
        if f"{bodypart}_x" not in df or f"{bodypart}_y" not in df:
            continue
        mask = _valid_bodypart_mask(df, bodypart)
        if not mask.any():
            continue
        used_bodyparts.append(bodypart)
        x_series.append(df[f"{bodypart}_x"].where(mask))
        y_series.append(df[f"{bodypart}_y"].where(mask))

    if not used_bodyparts:
        raise ValueError(
            "No bodyparts passed the likelihood threshold. Lower the threshold or inspect the DLC output."
        )

    centroid = pd.DataFrame(
        {
            "x": pd.concat(x_series, axis=1).mean(axis=1),
            "y": pd.concat(y_series, axis=1).mean(axis=1),
        }
    )
    centroid["valid_pose"] = centroid["x"].notna() & centroid["y"].notna()
    centroid["x"] = centroid["x"].interpolate(limit_direction="both")
    centroid["y"] = centroid["y"].interpolate(limit_direction="both")
    centroid["x_smooth"] = centroid["x"].rolling(window=5, min_periods=1, center=True).median()
    centroid["y_smooth"] = centroid["y"].rolling(window=5, min_periods=1, center=True).median()
    return centroid, used_bodyparts


def _smooth_series(series: pd.Series) -> pd.Series:
    return series.interpolate(limit_direction="both").rolling(window=5, min_periods=1, center=True).median()


def _bodypart_xy(df: pd.DataFrame, bodypart: str) -> tuple[pd.Series, pd.Series]:
    mask = _valid_bodypart_mask(df, bodypart)
    if not mask.any():
        empty = pd.Series(np.nan, index=df.index, dtype=float)
        return empty, empty
    x = _smooth_series(df[f"{bodypart}_x"].where(mask))
    y = _smooth_series(df[f"{bodypart}_y"].where(mask))
    return x, y


def _vector_angle_degrees(dx: pd.Series, dy: pd.Series) -> pd.Series:
    angle = np.degrees(np.arctan2(dy, dx))
    return pd.Series(angle, index=dx.index).replace([np.inf, -np.inf], np.nan)


def _angle_change_degrees(angle_series: pd.Series) -> pd.Series:
    radians = np.radians(angle_series)
    unwrapped = np.unwrap(radians.to_numpy())
    diffs = np.diff(unwrapped, prepend=unwrapped[0])
    return pd.Series(np.abs(np.degrees(diffs)), index=angle_series.index).fillna(0.0)


def _pair_distance(df: pd.DataFrame, point_a: str, point_b: str) -> pd.Series:
    if f"{point_a}_x" not in df or f"{point_b}_x" not in df:
        return pd.Series(np.nan, index=df.index, dtype=float)
    ax, ay = _bodypart_xy(df, point_a)
    bx, by = _bodypart_xy(df, point_b)
    return np.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _bool_segments(mask: pd.Series) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    start = None

    for idx, active in enumerate(mask.fillna(False).tolist()):
        if active and start is None:
            start = idx
        elif not active and start is not None:
            segments.append((start, idx - 1))
            start = None

    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def _dominant_behavior(metrics: dict[str, float]) -> str:
    for label in POSE_BEHAVIOR_PRIORITY:
        if metrics.get(f"{label}_ratio", 0.0) >= 0.12:
            return label
    return "monitoring"


def summarize_pose_analysis(
    df: pd.DataFrame,
    fps: float | None = None,
) -> dict:
    flattened = flatten_dlc_columns(df)
    centroid, used_bodyparts = compute_centroid_track(flattened)
    frame_count = len(centroid)

    dt = 1.0 / fps if fps and fps > 0 else 1.0
    timestamps = np.arange(frame_count, dtype=float) * dt

    dx = centroid["x_smooth"].diff().fillna(0.0)
    dy = centroid["y_smooth"].diff().fillna(0.0)
    displacement = np.sqrt(dx ** 2 + dy ** 2)
    speed = displacement / dt
    acceleration = speed.diff().fillna(0.0) / dt

    rolling_window = max(5, int(round(2.0 / dt)))
    rolling_distance = displacement.rolling(window=rolling_window, min_periods=1).sum()
    rolling_x_span = (
        centroid["x_smooth"].rolling(window=rolling_window, min_periods=1).max()
        - centroid["x_smooth"].rolling(window=rolling_window, min_periods=1).min()
    )
    rolling_y_span = (
        centroid["y_smooth"].rolling(window=rolling_window, min_periods=1).max()
        - centroid["y_smooth"].rolling(window=rolling_window, min_periods=1).min()
    )
    rolling_span = np.sqrt((rolling_x_span ** 2) + (rolling_y_span ** 2)).replace(0, np.nan)

    if {"nose", "tail_base"}.issubset(used_bodyparts):
        nose_x, nose_y = _bodypart_xy(flattened, "nose")
        tail_base_x, tail_base_y = _bodypart_xy(flattened, "tail_base")
        body_axis_angle = _vector_angle_degrees(nose_x - tail_base_x, nose_y - tail_base_y)
        body_length = np.sqrt((nose_x - tail_base_x) ** 2 + (nose_y - tail_base_y) ** 2)
    else:
        body_axis_angle = _vector_angle_degrees(dx, dy)
        body_length = pd.Series(np.nan, index=flattened.index, dtype=float)

    turning_rate = _angle_change_degrees(body_axis_angle) / max(dt, 1e-6)

    forepaw_span = _pair_distance(flattened, "left_forepaw", "right_forepaw")
    hindpaw_span = _pair_distance(flattened, "left_hindpaw", "right_hindpaw")
    tail_extension = _pair_distance(flattened, "tail_base", "tail_tip")
    ear_span = _pair_distance(flattened, "left_ear", "right_ear")

    grooming_local_motion = (
        forepaw_span.diff().abs().fillna(0.0)
        + hindpaw_span.diff().abs().fillna(0.0)
        + ear_span.diff().abs().fillna(0.0)
    ) / max(dt, 1e-6)

    torso_points = [point for point in ("upper_back", "spine_mid", "lower_back", "tail_base") if point in used_bodyparts]
    curvature_components: list[pd.Series] = []
    for start, mid, end in (("nose", "upper_back", "lower_back"), ("upper_back", "lower_back", "tail_base"), ("lower_back", "tail_base", "tail_tip")):
        if {start, mid, end}.issubset(used_bodyparts):
            sx, sy = _bodypart_xy(flattened, start)
            mx, my = _bodypart_xy(flattened, mid)
            ex, ey = _bodypart_xy(flattened, end)
            angle_a = np.arctan2(my - sy, mx - sx)
            angle_b = np.arctan2(ey - my, ex - mx)
            component = pd.Series(np.abs(np.degrees(np.unwrap((angle_b - angle_a).to_numpy()))), index=flattened.index)
            curvature_components.append(component)
    body_curvature = (
        pd.concat(curvature_components, axis=1).mean(axis=1)
        if curvature_components
        else pd.Series(0.0, index=flattened.index)
    ).fillna(0.0)

    moving_mask = speed >= MOVEMENT_SPEED_THRESHOLD
    immobile_mask = speed <= IMMOBILITY_SPEED_THRESHOLD
    turning_mask = (turning_rate >= 120.0) & (speed >= 8.0)
    confined_mask = rolling_span.fillna(0.0) <= 65.0
    stereotypy_mask = (rolling_distance >= 80.0) & confined_mask & (turning_rate >= 55.0)
    exploratory_mask = moving_mask & (rolling_span.fillna(0.0) >= 70.0) & ~stereotypy_mask
    locomotor_burst_mask = (speed >= 55.0) & (rolling_span.fillna(0.0) >= 80.0)
    freezing_mask = immobile_mask & (rolling_span.fillna(0.0) <= 20.0)
    hesitation_mask = (speed <= 12.0) & (speed.rolling(window=rolling_window, min_periods=1).mean() >= 18.0)
    grooming_mask = (
        (speed <= 12.0)
        & (rolling_span.fillna(0.0) <= 32.0)
        & ((grooming_local_motion >= 8.0) | (body_curvature >= 18.0))
    )
    resting_mask = immobile_mask & ~freezing_mask

    label_masks = {
        "immobility_bout": immobile_mask,
        "exploratory_locomotion": exploratory_mask,
        "stereotypy_candidate": stereotypy_mask,
        "freezing_candidate": freezing_mask,
        "grooming_candidate": grooming_mask,
        "turning_pattern": turning_mask,
        "locomotor_burst": locomotor_burst_mask,
        "hesitation": hesitation_mask,
        "resting": resting_mask,
        "exploration": exploratory_mask,
    }

    bouts: list[BoutRecord] = []
    for label, mask in label_masks.items():
        for start_idx, end_idx in _bool_segments(mask):
            duration_s = (end_idx - start_idx + 1) * dt
            if duration_s < MIN_BOUT_DURATION_S:
                continue
            segment_speed = speed.iloc[start_idx : end_idx + 1]
            confidence = float(mask.iloc[start_idx : end_idx + 1].mean())
            bouts.append(
                BoutRecord(
                    label=label,
                    start_s=float(timestamps[start_idx]),
                    end_s=float(timestamps[end_idx]),
                    duration_s=float(duration_s),
                    mean_speed_px_s=float(segment_speed.mean()),
                    confidence=confidence,
                )
            )

    total_distance = float(displacement.sum())
    duration_s = float(frame_count * dt)
    x_span = float(centroid["x_smooth"].max() - centroid["x_smooth"].min())
    y_span = float(centroid["y_smooth"].max() - centroid["y_smooth"].min())
    spatial_span = float(np.sqrt(x_span ** 2 + y_span ** 2))
    valid_pose_ratio = float(centroid["valid_pose"].mean())
    available_pairs = available_skeleton_pairs(used_bodyparts)
    expected_bodyparts = RAT_BODYPOINT_PRIORITY
    missing_priority_bodyparts = [part for part in expected_bodyparts if part not in used_bodyparts]

    metrics = {
        "duration_s": duration_s,
        "distance_px": total_distance,
        "mean_speed_px_s": float(speed.mean()),
        "peak_speed_px_s": float(speed.max()),
        "mean_acceleration_px_s2": float(acceleration.abs().mean()),
        "movement_ratio": float(moving_mask.mean()),
        "immobility_ratio": float(immobile_mask.mean()),
        "exploratory_ratio": float(exploratory_mask.mean()),
        "stereotypy_candidate_ratio": float(stereotypy_mask.mean()),
        "freezing_candidate_ratio": float(freezing_mask.mean()),
        "grooming_candidate_ratio": float(grooming_mask.mean()),
        "turning_pattern_ratio": float(turning_mask.mean()),
        "locomotor_burst_ratio": float(locomotor_burst_mask.mean()),
        "hesitation_ratio": float(hesitation_mask.mean()),
        "resting_ratio": float(resting_mask.mean()),
        "pace_confined_score": float((rolling_distance / rolling_span).replace([np.inf, -np.inf], np.nan).fillna(0).mean()),
        "spatial_span_px": spatial_span,
        "valid_pose_ratio": valid_pose_ratio,
        "mean_turning_rate_deg_s": float(turning_rate.mean()),
        "peak_turning_rate_deg_s": float(turning_rate.max()),
        "mean_body_length_px": float(body_length.mean()) if body_length.notna().any() else 0.0,
        "mean_tail_extension_px": float(tail_extension.mean()) if tail_extension.notna().any() else 0.0,
        "mean_body_curvature_deg": float(body_curvature.mean()),
        "grooming_motion_score": float(grooming_local_motion.mean()),
        "tracked_bodyparts_count": float(len(used_bodyparts)),
        "missing_priority_bodyparts_count": float(len(missing_priority_bodyparts)),
    }

    notes = [
        "Outputs are heuristic screening metrics derived from DeepLabCut pose traces, not a clinical or behavioral diagnosis.",
        "The stereotypy signal is designed to surface confined, repetitive, high-motion bouts for human review.",
        "The rat pose analyzer can exploit a richer body schema when ears, paws, spine points, and additional tail landmarks are present.",
    ]
    if fps is None:
        notes.append("FPS was not supplied, so timestamps are reported in frame units treated as seconds.")
    if valid_pose_ratio < 0.8:
        notes.append("Pose confidence was limited for part of the recording; inspect the DLC labels before interpreting subtle effects.")
    if len(used_bodyparts) <= 5:
        notes.append(
            "The current DeepLabCut model tracks only a compact 5-point rat skeleton, so paw- and ear-driven behaviors such as grooming and hesitation are less reliable than they would be with a richer retrained model."
        )

    return {
        "bodyparts_used": used_bodyparts,
        "expected_bodyparts": expected_bodyparts,
        "missing_priority_bodyparts": missing_priority_bodyparts,
        "skeleton_pairs": available_pairs,
        "dominant_behavior": _dominant_behavior(metrics),
        "metrics": metrics,
        "bouts": [bout.__dict__ for bout in sorted(bouts, key=lambda item: item.start_s)],
        "notes": notes,
    }


def analyze_dlc_csv(csv_path: str, fps: float | None = None) -> dict:
    df = load_dlc_csv(csv_path)
    return summarize_pose_analysis(df, fps=fps)
