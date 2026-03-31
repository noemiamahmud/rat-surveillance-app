# backend/app/behavior.py
import time
from dataclasses import dataclass, field
from typing import List, Optional

import cv2
import numpy as np

from .schemas import BehaviorEvent, BehaviorTimeline


@dataclass
class BehaviorAnalyzer:
    """Stub behavior analyzer.
    - Uses simple frame differencing as a stand-in for DLC.
    - Stores events with timestamps.
    """

    session_id: str
    _last_gray: Optional[np.ndarray] = None
    _events: List[BehaviorEvent] = field(default_factory=list)
    _last_state: str = "idle"
    _start_time: float = field(default_factory=time.time)

    def process_frame(self, frame_bgr) -> None:
        """Process one frame and maybe create a behavior event."""
        if frame_bgr is None:
            return

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if self._last_gray is None:
            self._last_gray = gray
            return

        diff = cv2.absdiff(self._last_gray, gray)
        _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        motion_score = float(np.mean(thresh))

        self._last_gray = gray

        # Very dumb logic: > threshold = "active", else "idle"
        new_state = "rat_active" if motion_score > 5.0 else "rat_idle"

        if new_state != self._last_state:
            ts = time.time() - self._start_time
            self._events.append(
                BehaviorEvent(
                    timestamp_s=ts,
                    label=new_state,
                    confidence=min(1.0, motion_score / 255.0),
                )
            )
            self._last_state = new_state

    def get_timeline(self) -> BehaviorTimeline:
        return BehaviorTimeline(session_id=self.session_id, events=self._events.copy())
