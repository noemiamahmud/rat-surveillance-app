# backend/app/dlc_runner.py
from __future__ import annotations
import importlib.util
from pathlib import Path
from typing import Optional

import cv2
import pandas as pd
import deeplabcut
import yaml

from .config import settings
from .pose_analysis import flatten_dlc_columns


def _ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_video_fps(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    if fps <= 0:
        raise RuntimeError(f"Invalid FPS read from video: {video_path}")
    return fps


def _find_dlc_csv_for_video(video_path: str) -> Path:
    """
    Given a video, find the DLC CSV file produced by analyze_videos.
    We look in the same folder and inside DLC_OUTPUT_ROOT if used.
    """
    video_path = Path(video_path)
    stem = video_path.stem

    candidates = []

    # search in same directory
    for p in video_path.parent.glob(f"{stem}*DLC*.csv"):
        candidates.append(p)

    # search in output root (e.g., if DLC project writes there)
    out_root = Path(settings.DLC_OUTPUT_ROOT)
    if out_root.exists():
        for p in out_root.rglob(f"{stem}*DLC*.csv"):
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError(
            f"No DLC CSV found for {video_path}. "
            f"Check that analyze_videos ran successfully."
        )

    # pick the most recently modified
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def _read_project_engine(config_path: str) -> str:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return str(config.get("engine", "tensorflow")).strip().lower()


def _validate_runtime_for_project(config_path: str) -> None:
    engine = _read_project_engine(config_path)
    if engine != "pytorch":
        return

    pytorch_pose_spec = importlib.util.find_spec("deeplabcut.pose_estimation_pytorch")
    if pytorch_pose_spec is None:
        raise RuntimeError(
            "This DeepLabCut project is configured for the PyTorch engine, but the installed DLC runtime "
            "does not provide the PyTorch pose inference module. The current environment can import DLC, "
            "but it only exposes the older TensorFlow-style analyze_videos path. To run this project in the app, "
            "install a DLC runtime with PyTorch inference support or retrain/export a TensorFlow-compatible model."
        )


def run_dlc_on_video(
    video_path: str,
    config_path: Optional[str] = None,
    save_as_csv: bool = True,
) -> pd.DataFrame:
    """
    Run DeepLabCut on the given video and return a pandas DataFrame with:
    - frame (int)
    - timestamp_s (float)
    - bodypart_coord columns (e.g. nose_x, nose_y, nose_likelihood, etc.)
    """
    if config_path is None:
        config_path = settings.DLC_CONFIG_PATH

    if not Path(config_path).exists():
        raise FileNotFoundError(
            f"DLC config not found at {config_path}. "
            f"Set DLC_CONFIG_PATH env var or config.DLC_CONFIG_PATH."
        )

    _validate_runtime_for_project(config_path)

    video_path = str(Path(video_path).resolve())
    _ensure_output_dir(Path(settings.DLC_OUTPUT_ROOT))

    # Run DLC (this writes H5/CSV to disk)
    deeplabcut.analyze_videos(
        config_path,
        [video_path],
        save_as_csv=save_as_csv,
        destfolder=str(settings.DLC_OUTPUT_ROOT),
    )

    # Find the CSV we just produced
    csv_path = _find_dlc_csv_for_video(video_path)

    # Load DLC results
    df = pd.read_csv(csv_path, header=[0, 1, 2])  # MultiIndex header
    df = flatten_dlc_columns(df)

    # add frame index and timestamps
    df["frame"] = df.index
    fps = _get_video_fps(video_path)
    df["timestamp_s"] = df["frame"] / fps

    return df
