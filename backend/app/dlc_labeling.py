from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import deeplabcut
from deeplabcut.gui.tabs.label_frames import label_frames

def _resolve_config_path(config_path: str | None) -> Path:
    if config_path:
        path = Path(config_path).expanduser().resolve()
    else:
        repo_root = Path(__file__).resolve().parent.parent.parent
        dlc_project_dir = repo_root / "dlc_project"
        direct_config = dlc_project_dir / "config.yaml"
        if direct_config.exists():
            path = direct_config
        else:
            nested_configs = sorted(dlc_project_dir.glob("*/config.yaml"))
            if not nested_configs:
                raise FileNotFoundError(f"No DLC config found under {dlc_project_dir}")
            path = nested_configs[0].resolve()
    if not path.exists():
        raise FileNotFoundError(f"DLC config not found: {path}")
    return path


def _resolve_videos(videos: Iterable[str]) -> list[str]:
    resolved: list[str] = []
    for video in videos:
        path = Path(video).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Video not found: {path}")
        resolved.append(str(path))
    if not resolved:
        raise ValueError("At least one video path is required.")
    return resolved


def _image_folder_name(video_path: str) -> str:
    return Path(video_path).stem


def add_videos(config_path: str | None, videos: Iterable[str], *, copy_videos: bool = False) -> list[str]:
    config = _resolve_config_path(config_path)
    resolved_videos = _resolve_videos(videos)
    deeplabcut.add_new_videos(str(config), resolved_videos, copy_videos=copy_videos)
    return resolved_videos


def extract_frames(
    config_path: str | None,
    videos: Iterable[str],
    *,
    mode: str = "automatic",
    algo: str = "kmeans",
    userfeedback: bool = False,
) -> list[str]:
    config = _resolve_config_path(config_path)
    resolved_videos = _resolve_videos(videos)
    deeplabcut.extract_frames(
        str(config),
        mode=mode,
        algo=algo,
        userfeedback=userfeedback,
        videos_list=resolved_videos,
    )
    return [_image_folder_name(video) for video in resolved_videos]


def open_labeler(config_path: str | None, image_folder: str | None = None) -> None:
    config = _resolve_config_path(config_path)
    label_frames(config_path=str(config), image_folder=image_folder)


def check_labels(config_path: str | None) -> None:
    config = _resolve_config_path(config_path)
    deeplabcut.check_labels(str(config))


def prepare_video(
    config_path: str | None,
    video_path: str,
    *,
    copy_videos: bool = False,
    mode: str = "automatic",
    algo: str = "kmeans",
    userfeedback: bool = False,
    label_now: bool = False,
) -> str:
    resolved_video = add_videos(config_path, [video_path], copy_videos=copy_videos)[0]
    image_folder = extract_frames(
        config_path,
        [resolved_video],
        mode=mode,
        algo=algo,
        userfeedback=userfeedback,
    )[0]
    if label_now:
        open_labeler(config_path, image_folder=image_folder)
    return image_folder


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Helpers for preparing and labeling DeepLabCut project videos."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Optional DLC config path. Defaults to backend/app/config.py settings.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    add_parser = subparsers.add_parser("add-video", help="Add one or more videos to the DLC project.")
    add_parser.add_argument("videos", nargs="+", help="Path(s) to MP4 or other DLC-supported video files.")
    add_parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="Copy the videos into the DLC project instead of referencing them in place.",
    )

    extract_parser = subparsers.add_parser("extract-frames", help="Extract labeling frames for one or more videos.")
    extract_parser.add_argument("videos", nargs="+", help="Path(s) to videos already added to the project.")
    extract_parser.add_argument("--mode", default="automatic", choices=["automatic", "manual"])
    extract_parser.add_argument("--algo", default="kmeans", choices=["kmeans", "uniform"])
    extract_parser.add_argument(
        "--userfeedback",
        action="store_true",
        help="Enable DLC's interactive extraction prompts.",
    )

    prepare_parser = subparsers.add_parser(
        "prepare-video",
        help="Add a video to the project and extract frames in one step.",
    )
    prepare_parser.add_argument("video", help="Path to a video file.")
    prepare_parser.add_argument("--mode", default="automatic", choices=["automatic", "manual"])
    prepare_parser.add_argument("--algo", default="kmeans", choices=["kmeans", "uniform"])
    prepare_parser.add_argument("--copy-videos", action="store_true")
    prepare_parser.add_argument("--userfeedback", action="store_true")
    prepare_parser.add_argument(
        "--label-now",
        action="store_true",
        help="Open the DLC labeling GUI immediately after extraction.",
    )

    label_parser = subparsers.add_parser("label", help="Open the DLC labeling GUI.")
    label_parser.add_argument(
        "--image-folder",
        default=None,
        help="Optional labeled-data subfolder name. Defaults to the first available folder in the project.",
    )

    subparsers.add_parser("check-labels", help="Render DLC's label-check images for the whole project.")

    args = parser.parse_args()

    if args.command == "add-video":
        added = add_videos(args.config, args.videos, copy_videos=args.copy_videos)
        print("Added videos:")
        for video in added:
            print(f"- {video}")
        return

    if args.command == "extract-frames":
        folders = extract_frames(
            args.config,
            args.videos,
            mode=args.mode,
            algo=args.algo,
            userfeedback=args.userfeedback,
        )
        print("Extracted frame folders:")
        for folder in folders:
            print(f"- {folder}")
        return

    if args.command == "prepare-video":
        folder = prepare_video(
            args.config,
            args.video,
            copy_videos=args.copy_videos,
            mode=args.mode,
            algo=args.algo,
            userfeedback=args.userfeedback,
            label_now=args.label_now,
        )
        print(f"Prepared labeling folder: {folder}")
        return

    if args.command == "label":
        open_labeler(args.config, image_folder=args.image_folder)
        return

    check_labels(args.config)


if __name__ == "__main__":
    main()
