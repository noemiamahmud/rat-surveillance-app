"""
CPU-first temporal classifier for laptop / limited-video lab setups.

The default production path no longer depends on a trained LSTM checkpoint.
Instead it:

1. Normalizes motion to the camera frame and to this session's own percentiles
   (so a 240p MacBook clip and a 1080p arena video share the same thresholds).
2. Scores all eight behaviors as soft emissions, optionally shifted by drug-condition
   priors and per-frame DeepLabCut pose cues.
3. Decodes the most likely label sequence with a Hidden Markov Model (Viterbi),
   which suppresses one-frame flicker without a GPU.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .condition_profiles import class_log_bias, condition_stay_boost
from .temporal_classifier import BEHAVIOR_LABELS, LABEL_TO_IDX, NUM_CLASSES, NUM_FEATURES


EPS = 1e-6
STAY_PROB = 0.90


def _sigmoid(x: np.ndarray | float, center: float, scale: float) -> np.ndarray:
    z = np.clip((np.asarray(x, dtype=np.float64) - center) / max(scale, EPS), -40, 40)
    return 1.0 / (1.0 + np.exp(-z))


def _peak(x: np.ndarray | float, mu: float, sigma: float) -> np.ndarray:
    z = (np.asarray(x, dtype=np.float64) - mu) / max(sigma, EPS)
    return np.exp(-0.5 * z * z)


def _high(x: np.ndarray | float, thresh: float, scale: float) -> np.ndarray:
    return _sigmoid(x, thresh, scale)


def _low(x: np.ndarray | float, thresh: float, scale: float) -> np.ndarray:
    return _sigmoid(-np.asarray(x, dtype=np.float64), -thresh, scale)


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    if n == 0:
        return values.astype(np.float32)
    window = max(1, min(int(window), n))
    c = np.cumsum(values, dtype=np.float64)
    out = np.empty(n, dtype=np.float64)
    out[:window] = c[:window] / np.arange(1, window + 1)
    if n > window:
        out[window:] = (c[window:] - c[:-window]) / window
    return out.astype(np.float32)


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    mean = _rolling_mean(values, window)
    mean_sq = _rolling_mean(values * values, window)
    return np.sqrt(np.maximum(mean_sq - mean * mean, 0.0)).astype(np.float32)


def _median_smooth(values: np.ndarray, kernel: int = 5) -> np.ndarray:
    if len(values) == 0:
        return values
    kernel = max(1, kernel | 1)
    try:
        from scipy.ndimage import median_filter

        return median_filter(values, size=kernel, mode="nearest").astype(np.float32)
    except Exception:
        pad = kernel // 2
        padded = np.pad(values, (pad, pad), mode="edge")
        windows = np.lib.stride_tricks.sliding_window_view(padded, kernel)
        return np.median(windows, axis=1).astype(np.float32)


def _log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=axis, keepdims=True) + EPS)


def session_normalization(
    features: np.ndarray,
    frame_width: int,
    frame_height: int,
) -> dict[str, np.ndarray | float]:
    """Resolution- and session-adaptive feature scaling."""
    diag = float(np.hypot(max(frame_width, 1), max(frame_height, 1)))
    speed = features[:, 0]
    motion = features[:, 1]
    span = features[:, 3]
    direction = features[:, 2]
    pace = features[:, 5]
    rolling_speed = features[:, 9]
    rolling_dir = features[:, 11]
    rolling_span = features[:, 12]

    p90_speed = float(max(np.percentile(speed, 90) if len(speed) else 1.0, 0.12 * diag, 1.0))
    p90_motion = float(max(np.percentile(motion, 90) if len(motion) else 1.0, 8.0, 1.0))
    p90_span = float(max(np.percentile(span, 90) if len(span) else 1.0, 0.22 * diag, 1.0))
    p90_turn = float(max(np.percentile(np.abs(direction), 90) if len(direction) else 1.0, 25.0, 1.0))

    geom_speed = speed / (0.28 * diag + EPS)
    session_speed = speed / (p90_speed + EPS)
    speed_n = np.clip(0.7 * geom_speed + 0.3 * session_speed, 0.0, 4.0)

    geom_span = span / (0.55 * diag + EPS)
    session_span = span / (p90_span + EPS)
    span_n = np.clip(0.7 * geom_span + 0.3 * session_span, 0.0, 4.0)

    motion_n = np.clip(motion / (p90_motion + EPS), 0.0, 4.0)
    turn_n = np.clip(np.abs(direction) / (p90_turn + EPS), 0.0, 4.0)
    rolling_turn_n = np.clip(np.abs(rolling_dir) / 90.0, 0.0, 4.0)
    pace_n = np.clip(pace / 1.8, 0.0, 4.0)
    speed_drop = np.clip((rolling_speed - speed) / (p90_speed + EPS), 0.0, 4.0)
    rolling_span_n = np.clip(rolling_span / (p90_span + EPS), 0.0, 4.0)

    pose_curv = np.clip(features[:, 13] / 25.0, 0.0, 4.0)
    pose_turn = np.clip(features[:, 14] / 90.0, 0.0, 4.0)
    pose_groom = np.clip(features[:, 20] / 12.0, 0.0, 4.0)
    pose_present = float(np.any(np.abs(features[:, 13:22]) > 1e-6))

    return {
        "speed_n": speed_n.astype(np.float32),
        "span_n": span_n.astype(np.float32),
        "motion_n": motion_n.astype(np.float32),
        "turn_n": turn_n.astype(np.float32),
        "rolling_turn_n": rolling_turn_n.astype(np.float32),
        "pace_n": pace_n.astype(np.float32),
        "speed_drop": speed_drop.astype(np.float32),
        "rolling_span_n": rolling_span_n.astype(np.float32),
        "pose_curv": pose_curv.astype(np.float32),
        "pose_turn": pose_turn.astype(np.float32),
        "pose_groom": pose_groom.astype(np.float32),
        "pose_present": pose_present,
        "diag": diag,
        "p90_speed": p90_speed,
        "p90_span": p90_span,
    }


def emission_logits(
    features: np.ndarray,
    frame_width: int,
    frame_height: int,
    condition: str | None = None,
) -> tuple[np.ndarray, dict]:
    """Soft class scores for every frame. Shape (T, NUM_CLASSES)."""
    n = len(features)
    stats = session_normalization(features, frame_width, frame_height)
    speed_n = stats["speed_n"]
    span_n = stats["span_n"]
    motion_n = stats["motion_n"]
    turn_n = stats["turn_n"]
    rolling_turn_n = stats["rolling_turn_n"]
    pace_n = stats["pace_n"]
    speed_drop = stats["speed_drop"]
    rolling_span_n = stats["rolling_span_n"]
    pose_curv = stats["pose_curv"]
    pose_turn = stats["pose_turn"]
    pose_groom = stats["pose_groom"]
    pose_w = 0.55 if stats["pose_present"] else 0.0

    scores = np.zeros((n, NUM_CLASSES), dtype=np.float64)

    locomotor = _high(speed_n, 1.15, 0.22) * _high(span_n, 0.85, 0.20)
    exploration = _peak(speed_n, 0.72, 0.32) * _high(span_n, 0.48, 0.18) * _low(pace_n, 1.35, 0.25)
    freeze = _low(speed_n, 0.12, 0.06) * _low(motion_n, 0.18, 0.08) * _low(span_n, 0.22, 0.08)
    groom = (
        _low(speed_n, 0.28, 0.10)
        * _low(span_n, 0.28, 0.10)
        * _high(motion_n, 0.45, 0.15)
        * (1.0 + pose_w * _high(pose_groom, 0.45, 0.18) + pose_w * _high(pose_curv, 0.5, 0.2))
    )
    hesitation = _high(speed_drop, 0.28, 0.10) * _low(speed_n, 0.42, 0.12) * _high(motion_n, 0.22, 0.12)
    stereotypy = (
        _high(speed_n, 0.45, 0.14)
        * _low(span_n, 0.42, 0.12)
        * (_high(pace_n, 0.85, 0.18) + 0.6 * _high(rolling_turn_n, 0.7, 0.2) + pose_w * _high(pose_turn, 0.4, 0.2))
        * _low(rolling_span_n, 0.55, 0.15)
    )
    turning = (
        _high(speed_n, 0.32, 0.12)
        * _high(turn_n + rolling_turn_n + pose_w * pose_turn, 0.85, 0.22)
        * _low(span_n, 0.85, 0.20)
    )
    resting = _low(speed_n, 0.16, 0.07) * _low(motion_n, 0.32, 0.10) * _low(freeze, 0.55, 0.15)
    monitoring = 0.18 * np.ones(n, dtype=np.float64)

    scores[:, LABEL_TO_IDX["locomotor_burst"]] = locomotor
    scores[:, LABEL_TO_IDX["exploration"]] = exploration
    scores[:, LABEL_TO_IDX["freezing_candidate"]] = freeze
    scores[:, LABEL_TO_IDX["grooming_candidate"]] = groom
    scores[:, LABEL_TO_IDX["hesitation"]] = hesitation
    scores[:, LABEL_TO_IDX["stereotypy_candidate"]] = stereotypy
    scores[:, LABEL_TO_IDX["turning_pattern"]] = turning
    scores[:, LABEL_TO_IDX["resting"]] = resting
    scores[:, LABEL_TO_IDX["monitoring"]] = monitoring

    scores = np.clip(scores, 1e-4, None)
    logits = np.log(scores)
    logits = logits + class_log_bias(condition)
    return logits.astype(np.float32), stats


def transition_log_matrix(condition: str | None = None) -> np.ndarray:
    """Sticky, biologically plausible class-to-class transitions."""
    groups = {
        "immobile": {"freezing_candidate", "resting"},
        "local": {"grooming_candidate", "hesitation", "monitoring"},
        "travel": {"exploration", "locomotor_burst"},
        "repeat": {"stereotypy_candidate", "turning_pattern"},
    }
    label_group = {}
    for group_name, members in groups.items():
        for label in members:
            label_group[label] = group_name

    stay_boost = condition_stay_boost(condition)
    trans = np.full((NUM_CLASSES, NUM_CLASSES), 0.012, dtype=np.float64)

    for i, src in enumerate(BEHAVIOR_LABELS):
        stay = STAY_PROB + stay_boost.get(src, 0.0)
        stay = min(stay, 0.965)
        trans[i, :] = (1.0 - stay) / max(NUM_CLASSES - 1, 1)
        trans[i, i] = stay
        for j, dst in enumerate(BEHAVIOR_LABELS):
            if i == j:
                continue
            if label_group.get(src) == label_group.get(dst):
                trans[i, j] *= 2.4
            # Rapid freeze <-> sprint is rare
            if {src, dst} == {"freezing_candidate", "locomotor_burst"}:
                trans[i, j] *= 0.15
            if {src, dst} == {"resting", "locomotor_burst"}:
                trans[i, j] *= 0.25
        trans[i, :] /= trans[i, :].sum()

    return np.log(trans + EPS)


def viterbi(log_emissions: np.ndarray, log_trans: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Decode the most likely label path. Returns (path_idx, path_confidence)."""
    t_count, n_classes = log_emissions.shape
    if t_count == 0:
        return np.zeros(0, dtype=np.int32), np.zeros(0, dtype=np.float32)

    log_start = np.full(n_classes, -np.log(n_classes), dtype=np.float64)
    dp = np.full((t_count, n_classes), -np.inf, dtype=np.float64)
    back = np.zeros((t_count, n_classes), dtype=np.int32)
    dp[0] = log_start + log_emissions[0]

    for t in range(1, t_count):
        scores = dp[t - 1][:, None] + log_trans
        back[t] = scores.argmax(axis=0)
        dp[t] = scores.max(axis=0) + log_emissions[t]

    path = np.zeros(t_count, dtype=np.int32)
    path[-1] = int(dp[-1].argmax())
    for t in range(t_count - 1, 0, -1):
        path[t - 1] = back[t, path[t]]

    log_probs = _log_softmax(log_emissions, axis=1)
    conf = np.exp(log_probs[np.arange(t_count), path])
    # Reward temporally stable assignments
    stable = np.ones(t_count, dtype=np.float64)
    if t_count > 2:
        same_as_neighbors = (path == np.roll(path, 1)) & (path == np.roll(path, -1))
        stable = np.where(same_as_neighbors, 1.08, 0.92)
    conf = np.clip(conf * stable, 0.05, 0.99)
    return path, conf.astype(np.float32)


def stabilize_labels(labels: list[str], min_run: int) -> list[str]:
    """Absorb isolated blips shorter than min_run into the surrounding label."""
    n = len(labels)
    if n == 0 or min_run <= 1:
        return list(labels)
    out = list(labels)
    i = 0
    while i < n:
        j = i + 1
        while j < n and out[j] == out[i]:
            j += 1
        if (j - i) < min_run:
            left = out[i - 1] if i else None
            right = out[j] if j < n else None
            fill = left or right
            if fill is not None:
                out[i:j] = [fill] * (j - i)
        i = j
    return out


@dataclass
class SequenceClassification:
    labels: list[str]
    confidences: list[float]
    emissions: np.ndarray
    stats: dict


def classify_feature_sequence(
    features: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
    fps: float = 20.0,
    condition: str | None = None,
) -> SequenceClassification:
    """Full-session adaptive HMM classification."""
    if features.ndim != 2 or features.shape[1] != NUM_FEATURES:
        raise ValueError(f"Expected features of shape (T, {NUM_FEATURES}), got {features.shape}")

    smoothed = features.copy()
    for col in (0, 1, 3, 5):
        smoothed[:, col] = _median_smooth(smoothed[:, col], kernel=5)

    logits, stats = emission_logits(smoothed, frame_width, frame_height, condition=condition)
    path, conf = viterbi(logits, transition_log_matrix(condition))
    labels = [BEHAVIOR_LABELS[idx] for idx in path]
    min_run = max(3, int(round(0.18 * max(fps, 1.0))))
    labels = stabilize_labels(labels, min_run=min_run)
    stats["min_run"] = min_run
    stats["condition"] = condition or ""
    return SequenceClassification(
        labels=labels,
        confidences=[float(c) for c in conf],
        emissions=logits,
        stats={k: (float(v) if isinstance(v, (np.floating, float, int)) else v) for k, v in stats.items() if k in {"diag", "p90_speed", "p90_span", "pose_present", "min_run", "condition"}},
    )


class OnlineAdaptiveClassifier:
    """Forward-filter version of the HMM for live camera sessions."""

    def __init__(
        self,
        frame_width: int,
        frame_height: int,
        fps: float = 20.0,
        condition: str | None = None,
        window_seconds: float = 2.0,
    ):
        self.frame_width = max(int(frame_width), 1)
        self.frame_height = max(int(frame_height), 1)
        self.fps = max(float(fps), 1.0)
        self.condition = condition
        self.window = max(5, int(self.fps * window_seconds))
        self.speeds: deque[float] = deque(maxlen=240)
        self.motion: deque[float] = deque(maxlen=240)
        self.spans: deque[float] = deque(maxlen=240)
        self.dirs: deque[float] = deque(maxlen=240)
        self.pace: deque[float] = deque(maxlen=240)
        self.log_alpha = np.full(NUM_CLASSES, -np.log(NUM_CLASSES), dtype=np.float64)
        self.log_trans = transition_log_matrix(condition)
        self.last_label = "monitoring"
        self.last_confidence = 0.35

    def update(
        self,
        *,
        speed: float,
        motion_score: float,
        direction_delta: float,
        spatial_span: float,
        pace_confined_score: float,
    ) -> tuple[str, float]:
        self.speeds.append(float(speed))
        self.motion.append(float(motion_score))
        self.spans.append(float(spatial_span))
        self.dirs.append(float(direction_delta))
        self.pace.append(float(pace_confined_score))

        n = len(self.speeds)
        features = np.zeros((n, NUM_FEATURES), dtype=np.float32)
        features[:, 0] = np.fromiter(self.speeds, dtype=np.float32)
        features[:, 1] = np.fromiter(self.motion, dtype=np.float32)
        features[:, 2] = np.fromiter(self.dirs, dtype=np.float32)
        features[:, 3] = np.fromiter(self.spans, dtype=np.float32)
        features[:, 5] = np.fromiter(self.pace, dtype=np.float32)
        features[:, 9] = _rolling_mean(features[:, 0], self.window)
        features[:, 10] = _rolling_std(features[:, 0], self.window)
        features[:, 11] = _rolling_mean(features[:, 2], self.window)
        features[:, 12] = _rolling_mean(features[:, 3], self.window)

        logits, _ = emission_logits(
            features,
            self.frame_width,
            self.frame_height,
            condition=self.condition,
        )
        log_emit = _log_softmax(logits[-1])
        self.log_alpha = log_emit + np.logaddexp.reduce(self.log_alpha[:, None] + self.log_trans, axis=0)
        self.log_alpha -= np.logaddexp.reduce(self.log_alpha)
        idx = int(self.log_alpha.argmax())
        conf = float(np.clip(np.exp(self.log_alpha[idx]), 0.05, 0.99))
        self.last_label = BEHAVIOR_LABELS[idx]
        self.last_confidence = conf
        return self.last_label, self.last_confidence
