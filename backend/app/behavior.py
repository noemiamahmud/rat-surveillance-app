# backend/app/behavior.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .adaptive_classifier import OnlineAdaptiveClassifier
from .schemas import BehaviorEvent, BehaviorTimeline

WINDOW_SECONDS = 2.0


@dataclass
class BehaviorAnalyzer:
    """Live-session analyzer using the same adaptive HMM as offline upload analysis."""

    session_id: str
    condition: Optional[str] = None
    fps: float = 20.0
    _last_gray: Optional[np.ndarray] = None
    _events: List[BehaviorEvent] = field(default_factory=list)
    _last_state: str = "monitoring"
    _start_time: float = field(default_factory=time.time)
    _classifier: Optional[OnlineAdaptiveClassifier] = None
    _prev_centroid: Optional[tuple[float, float]] = None
    _prev_vector: Optional[np.ndarray] = None
    _centroids: list[tuple[float | None, float | None]] = field(default_factory=list)
    _speeds: list[float] = field(default_factory=list)

    def process_frame(self, frame_bgr) -> None:
        if frame_bgr is None:
            return

        height, width = frame_bgr.shape[:2]
        if self._classifier is None:
            self._classifier = OnlineAdaptiveClassifier(
                frame_width=width,
                frame_height=height,
                fps=self.fps,
                condition=self.condition,
            )

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        motion_score = 0.0
        centroid_x = None
        centroid_y = None
        if self._last_gray is not None:
            diff = cv2.absdiff(self._last_gray, gray)
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_score = float(np.mean(thresh))
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 120:
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        centroid_x = moments["m10"] / moments["m00"]
                        centroid_y = moments["m01"] / moments["m00"]

        self._last_gray = gray
        if (centroid_x is None or centroid_y is None) and self._prev_centroid is not None:
            centroid_x, centroid_y = self._prev_centroid

        speed = 0.0
        direction_delta = 0.0
        if centroid_x is not None and centroid_y is not None and self._prev_centroid is not None:
            current_vector = np.array(
                [centroid_x - self._prev_centroid[0], centroid_y - self._prev_centroid[1]],
                dtype=float,
            )
            speed = float(np.hypot(current_vector[0], current_vector[1])) * self.fps
            if (
                self._prev_vector is not None
                and np.linalg.norm(current_vector) > 0
                and np.linalg.norm(self._prev_vector) > 0
            ):
                cosine = np.clip(
                    float(
                        np.dot(current_vector, self._prev_vector)
                        / (np.linalg.norm(current_vector) * np.linalg.norm(self._prev_vector))
                    ),
                    -1.0,
                    1.0,
                )
                direction_delta = float(np.degrees(np.arccos(cosine)))
            if np.linalg.norm(current_vector) > 0:
                self._prev_vector = current_vector
        if centroid_x is not None and centroid_y is not None:
            self._prev_centroid = (centroid_x, centroid_y)

        self._centroids.append((centroid_x, centroid_y))
        self._speeds.append(speed)
        window_frames = max(5, int(self.fps * WINDOW_SECONDS))
        x_values = [x for x, _ in self._centroids[-window_frames:] if x is not None]
        y_values = [y for _, y in self._centroids[-window_frames:] if y is not None]
        spatial_span = (
            float(np.hypot(max(x_values) - min(x_values), max(y_values) - min(y_values)))
            if x_values and y_values
            else 0.0
        )
        recent_avg_speed = float(np.mean(self._speeds[-window_frames:])) if self._speeds else 0.0
        pace_confined_score = recent_avg_speed / max(spatial_span, 1.0)

        label, confidence = self._classifier.update(
            speed=speed,
            motion_score=motion_score,
            direction_delta=direction_delta,
            spatial_span=spatial_span,
            pace_confined_score=pace_confined_score,
        )
        if label != self._last_state:
            ts = time.time() - self._start_time
            self._events.append(
                BehaviorEvent(
                    timestamp_s=ts,
                    label=label,
                    confidence=confidence,
                )
            )
            self._last_state = label

    def get_timeline(self) -> BehaviorTimeline:
        return BehaviorTimeline(session_id=self.session_id, events=self._events.copy())
