# backend/app/video_stream.py
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict

import cv2

from .config import settings
from .behavior import BehaviorAnalyzer
from .schemas import SessionInfo


class SessionRecorder:
    """Handles writing frames to disk as a video file."""

    def __init__(self, session_id: str, name: Optional[str] = None):
        self.session_id = session_id
        self.name = name
        self.started_at = datetime.utcnow()
        safe_name = (name or "session").replace(" ", "_")
        filename = f"{self.started_at.strftime('%Y%m%d_%H%M%S')}_{safe_name}.mp4"
        self.output_path: Path = settings.RAW_VIDEOS_DIR / filename

        self._writer = None
        self._fps = 20.0  # target FPS
        self._fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._last_frame_time = 0.0

    def write_frame(self, frame):
        now = time.time()
        # Throttle to target FPS
        if now - self._last_frame_time < 1.0 / self._fps:
            return
        self._last_frame_time = now

        if frame is None:
            return

        h, w = frame.shape[:2]
        if self._writer is None:
            self._writer = cv2.VideoWriter(
                str(self.output_path), self._fourcc, self._fps, (w, h)
            )

        self._writer.write(frame)

    def close(self):
        if self._writer is not None:
            self._writer.release()
            self._writer = None


class VideoStreamManager:
    """Singleton-style manager for live RTSP, recording, and behavior."""

    def __init__(self):
        self.rtsp_url: Optional[str] = None
        self.capture: Optional[cv2.VideoCapture] = None
        self.running: bool = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._last_frame = None

        # Recording / behavior
        self._active_session: Optional[SessionRecorder] = None
        self._sessions: Dict[str, SessionInfo] = {}
        self._behavior_analyzer: Optional[BehaviorAnalyzer] = None

    # ---------- Camera control ----------

    def connect(self, rtsp_url: Optional[str] = None):
        if self.running:
            return

        self.rtsp_url = rtsp_url or settings.WYZE_RTSP_URL
        self.capture = cv2.VideoCapture(self.rtsp_url)

        if not self.capture.isOpened():
            self.capture.release()
            self.capture = None
            raise RuntimeError(f"Failed to open RTSP stream: {self.rtsp_url}")

        self.running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def disconnect(self):
        self.running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

        if self.capture:
            self.capture.release()
            self.capture = None

        self._last_frame = None

        # Stop any active session cleanly
        if self._active_session:
            self.stop_session()

    def _loop(self):
        while self.running and self.capture:
            ret, frame = self.capture.read()
            if not ret:
                time.sleep(0.1)
                continue

            with self._lock:
                self._last_frame = frame

            # Feed behavior analyzer
            if self._behavior_analyzer is not None:
                self._behavior_analyzer.process_frame(frame)

            # Write to recorder if active
            if self._active_session is not None:
                self._active_session.write_frame(frame)

        # Cleanup when loop exits
        if self.capture:
            self.capture.release()
            self.capture = None

    def get_jpeg_frame(self) -> Optional[bytes]:
        with self._lock:
            frame = None if self._last_frame is None else self._last_frame.copy()

        if frame is None:
            return None

        ok, jpeg = cv2.imencode(".jpg", frame)
        if not ok:
            return None
        return jpeg.tobytes()

    # ---------- Session control ----------

    def start_session(self, name: Optional[str] = None) -> SessionInfo:
        if not self.running:
            raise RuntimeError("Camera stream not running; connect first.")

        if self._active_session is not None:
            raise RuntimeError("A session is already active.")

        session_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        recorder = SessionRecorder(session_id=session_id, name=name)
        self._active_session = recorder
        self._behavior_analyzer = BehaviorAnalyzer(session_id=session_id)

        info = SessionInfo(
            session_id=session_id,
            name=name,
            started_at=recorder.started_at,
            ended_at=None,
            video_path=None,
        )
        self._sessions[session_id] = info
        return info

    def stop_session(self) -> Optional[SessionInfo]:
        if self._active_session is None:
            return None

        recorder = self._active_session
        recorder.close()

        # Update session info
        info = self._sessions.get(recorder.session_id)
        if info:
            info.ended_at = datetime.utcnow()
            info.video_path = str(recorder.output_path)

        self._persist_timeline(recorder.session_id)
        self._active_session = None
        self._behavior_analyzer = None
        return info

    def list_sessions(self) -> Dict[str, SessionInfo]:
        return self._sessions

    # ---------- Behavior timeline ----------

    def _timeline_path(self, session_id: str) -> Path:
        return settings.PROCESSED_DIR / "session_timelines" / f"{session_id}.json"

    def _persist_timeline(self, session_id: str) -> None:
        if self._behavior_analyzer and self._behavior_analyzer.session_id == session_id:
            import json
            timeline = self._behavior_analyzer.get_timeline()
            out = self._timeline_path(session_id)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(timeline.dict(), default=str), encoding="utf-8")

    def get_behavior_timeline(self, session_id: str):
        if (
            self._active_session
            and self._behavior_analyzer
            and self._active_session.session_id == session_id
        ):
            return self._behavior_analyzer.get_timeline()

        import json
        timeline_file = self._timeline_path(session_id)
        if timeline_file.exists():
            from .schemas import BehaviorTimeline
            data = json.loads(timeline_file.read_text(encoding="utf-8"))
            return BehaviorTimeline(**data)

        raise FileNotFoundError(
            f"No behavior timeline found for session {session_id}."
        )


# Global instance used by routes
stream_manager = VideoStreamManager()
