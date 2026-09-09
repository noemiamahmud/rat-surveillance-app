from __future__ import annotations

import json
import queue
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np

from .config import settings
from .observer_llm import llm_available, llm_descriptor, summarize_with_llm
from .adaptive_classifier import classify_feature_sequence
from .temporal_classifier import (
    get_classifier,
    extract_features_from_motion,
    enrich_with_pose,
)
from .simba_classifier import get_simba_classifier
from .condition_profiles import (
    condition_adjusted_review_priority,
    detect_anomalies,
    generate_condition_report,
)


WINDOW_SECONDS = 2.0
MIN_BOUT_SECONDS = 0.6


@dataclass
class FrameState:
    timestamp_s: float
    label: str
    confidence: float
    speed: float
    motion_score: float
    x: float | None
    y: float | None
    bbox_area_ratio: float
    direction_delta: float
    spatial_span: float
    pace_confined_score: float


def _transcode_for_web(input_path: Path, output_path: Path) -> Path:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(f"ffmpeg failed to transcode annotated video: {completed.stderr.strip()}")
    return output_path


def _segment_bouts(states: list[FrameState]) -> list[dict]:
    if not states:
        return []

    bouts: list[dict] = []
    start = 0
    for idx in range(1, len(states) + 1):
        boundary = idx == len(states) or states[idx].label != states[start].label
        if not boundary:
            continue

        segment = states[start:idx]
        duration = segment[-1].timestamp_s - segment[0].timestamp_s if len(segment) > 1 else 0.0
        if duration >= MIN_BOUT_SECONDS and segment[0].label != "monitoring":
            bouts.append(
                {
                    "label": segment[0].label,
                    "start_s": float(segment[0].timestamp_s),
                    "end_s": float(segment[-1].timestamp_s),
                    "duration_s": float(duration),
                    "mean_speed_px_s": float(np.mean([item.speed for item in segment])),
                    "confidence": float(np.mean([item.confidence for item in segment])),
                }
            )
        start = idx
    return bouts


def _generate_behavior_events(states: list[FrameState], bouts: list[dict]) -> list[dict]:
    events: list[dict] = []
    for bout in bouts:
        events.append(
            {
                "timestamp_s": bout["start_s"],
                "label": bout["label"],
                "confidence": bout["confidence"],
                "note": _note_for_label(bout["label"]),
            }
        )
    return events


def _note_for_label(label: str) -> str:
    return {
        "exploration": "The rat is covering space with sustained locomotion.",
        "locomotor_burst": "Rapid forward movement suggests a locomotor burst.",
        "freezing_candidate": "A low-motion pause suggests freezing or strong hesitation.",
        "hesitation": "Movement slowed abruptly, which may indicate hesitation.",
        "grooming_candidate": "Low displacement with local motion may reflect grooming-like behavior.",
        "stereotypy_candidate": "Confined repetitive motion suggests a stereotypy candidate.",
        "turning_pattern": "Repeated turning may indicate circling or biased movement.",
        "resting": "Sustained low motion suggests resting or immobility.",
    }.get(label, "Behavioral event detected.")


def _generate_analyst_messages(
    behavior_events: list[dict],
    dominant_behavior: str,
    pipeline_mode: str,
) -> list[dict]:
    messages = [
        {
            "timestamp_s": 0.0,
            "speaker": "AI Observer",
            "message": (
                f"Rat behavior analysis started in {pipeline_mode.replace('_', ' ')} mode. "
                f"I'll log notable behaviors with timestamps as they appear."
            ),
        }
    ]
    for event in behavior_events:
        messages.append(
            {
                "timestamp_s": event["timestamp_s"],
                "speaker": "AI Observer",
                "message": (
                    f"{event['label']} detected at {event['timestamp_s']:.2f}s "
                    f"with confidence {event['confidence']:.2f}. {event['note']}"
                ),
            }
        )

    messages.append(
        {
            "timestamp_s": behavior_events[-1]["timestamp_s"] if behavior_events else 0.0,
            "speaker": "AI Observer",
            "message": f"Session summary: dominant state was {dominant_behavior}.",
        }
    )
    return messages


def _build_analysis_summary(
    dominant_behavior: str,
    behavior_share: dict[str, float],
    bouts: list[dict],
    review_priority: str,
    pipeline_mode: str,
) -> str:
    top_bouts = Counter(bout["label"] for bout in bouts)
    bout_text = ", ".join(f"{count} {label}" for label, count in top_bouts.most_common(3)) or "no strong bouts"
    dominant_share = behavior_share.get(dominant_behavior, 0.0) * 100
    return (
        f"The rat behavior classifier marked {dominant_behavior} as the dominant state "
        f"({dominant_share:.1f}% of analyzed frames). Detected bouts included {bout_text}. "
        f"The pipeline ran in {pipeline_mode.replace('_', ' ')} mode and assigned a {review_priority.lower()} review priority."
    )


def _maybe_pose_enrichment(input_path: Path, fps: float) -> tuple[dict | None, str | None]:
    try:
        from .dlc_runner import run_dlc_on_video
        from .pose_analysis import summarize_pose_analysis
    except Exception as exc:
        return None, f"DeepLabCut backend import failed: {exc}"

    try:
        pose_df = run_dlc_on_video(video_path=str(input_path), save_as_csv=True)
        summary = summarize_pose_analysis(pose_df, fps=fps)
        keypoint_map: dict[int, dict[str, tuple[float, float, float]]] = {}
        bodyparts = summary.get("bodyparts_used", [])
        for idx, row in pose_df.iterrows():
            frame_points: dict[str, tuple[float, float, float]] = {}
            for bodypart in bodyparts:
                x = float(row.get(f"{bodypart}_x", np.nan))
                y = float(row.get(f"{bodypart}_y", np.nan))
                likelihood = float(row.get(f"{bodypart}_likelihood", 0.0))
                if not np.isnan(x) and not np.isnan(y):
                    frame_points[bodypart] = (x, y, likelihood)
            if frame_points:
                keypoint_map[int(idx)] = frame_points
        return {"summary": summary, "frame_keypoints": keypoint_map, "pose_df": pose_df}, None
    except Exception as exc:
        return None, str(exc)


def _selected_models(pose_enrichment: dict | None, classifier_mode: str = "heuristic") -> dict[str, str]:
    pose_bodyparts = []
    if pose_enrichment is not None:
        pose_bodyparts = pose_enrichment.get("summary", {}).get("bodyparts_used", [])

    if classifier_mode == "pytorch_temporal":
        temporal_desc = "PyTorch temporal LSTM/Transformer classifier (trained on labeled behavior sequences)"
    elif classifier_mode == "simba":
        temporal_desc = "SimBA LSTM behavioral classifier (pose-sequence supervised learning)"
    else:
        temporal_desc = (
            "Adaptive HMM classifier (session-normalized motion, condition priors, "
            "Viterbi temporal decoding; runs on CPU without a trained checkpoint)"
        )

    return {
        "pose_model": (
            "DeepLabCut PyTorch pose model with ResNet-50 GN backbone and heatmap head "
            f"(tracked bodyparts this run: {', '.join(pose_bodyparts)})"
            if pose_enrichment is not None
            else "No pose model was used in this run; motion fallback path was selected"
        ),
        "temporal_behavior_model": temporal_desc,
        "agent_model": (
            f"LLM-backed observer summary using {llm_descriptor()}" if llm_available()
            else "AI Observer event interpreter v0 (rule-based timestamped message generator)"
        ),
        "embedding_model": "None",
        "llm_model": llm_descriptor() if llm_available() else "None",
    }


def _pose_runtime_status() -> tuple[bool, str, str]:
    if not Path(settings.DLC_CONFIG_PATH).exists():
        return (
            False,
            "motion_fallback",
            f"DeepLabCut config was not found at {settings.DLC_CONFIG_PATH}.",
        )
    try:
        from .dlc_runner import run_dlc_on_video  # noqa: F401
    except Exception as exc:
        return (
            False,
            "motion_fallback",
            f"DeepLabCut import failed inside the backend runtime: {exc}",
        )
    return True, "attempting_deeplabcut_pose", f"Using DLC config at {settings.DLC_CONFIG_PATH}."


def _draw_overlay(frame, state: FrameState, dominant_behavior: str) -> np.ndarray:
    output = frame.copy()
    _, w = output.shape[:2]

    cv2.rectangle(output, (0, 0), (w, 96), (14, 25, 34), -1)
    cv2.putText(
        output,
        f"Rat behavior: {state.label}",
        (18, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.86,
        (240, 248, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Confidence {state.confidence:.2f}   Speed {state.speed:.1f}px/s   Motion {state.motion_score:.1f}",
        (18, 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (188, 220, 242),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        f"Session trend: {dominant_behavior}   Span {state.spatial_span:.1f}px   Pace score {state.pace_confined_score:.2f}",
        (18, 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (148, 196, 124),
        1,
        cv2.LINE_AA,
    )

    if state.x is not None and state.y is not None:
        cv2.circle(output, (int(state.x), int(state.y)), 8, (41, 196, 255), 2)
    return output


def _draw_pose_overlay(frame, pose_enrichment: dict | None, frame_index: int) -> np.ndarray:
    if pose_enrichment is None:
        return frame
    frame_pose = pose_enrichment.get("frame_keypoints", {}).get(frame_index)
    if not frame_pose:
        return frame

    output = frame.copy()
    skeleton_pairs = pose_enrichment.get("summary", {}).get("skeleton_pairs", [])
    for bodypart, point in frame_pose.items():
        x, y, likelihood = point
        if likelihood < 0.5:
            continue
        cv2.circle(output, (int(x), int(y)), 5, (255, 208, 0), -1)
        cv2.putText(
            output,
            bodypart,
            (int(x) + 6, int(y) - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (255, 208, 0),
            1,
            cv2.LINE_AA,
        )
    for start, end in skeleton_pairs:
        if start in frame_pose and end in frame_pose:
            sx, sy, sl = frame_pose[start]
            ex, ey, el = frame_pose[end]
            if sl >= 0.5 and el >= 0.5:
                cv2.line(output, (int(sx), int(sy)), (int(ex), int(ey)), (76, 220, 180), 2)
    return output


def _run_behavior_analysis(
    video_path: str,
    output_root: str,
    stream_callback=None,
    condition: str | None = None,
    classifier_type: str | None = None,
) -> dict:
    input_path = Path(video_path).expanduser().resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"Video not found: {input_path}")

    analysis_id = uuid4().hex[:10]
    artifact_dir = Path(output_root).expanduser().resolve() / analysis_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {input_path}")

    fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
    if fps <= 0:
        fps = 20.0
    pose_runtime_available, initial_pipeline_mode, pose_runtime_message = _pose_runtime_status()

    requested_classifier = (classifier_type or settings.CLASSIFIER_TYPE or "adaptive_hmm").strip().lower()
    if requested_classifier not in {"heuristic", "adaptive_hmm", "pytorch_temporal", "simba"}:
        requested_classifier = "adaptive_hmm"

    nn_classifier = get_classifier(requested_classifier)
    simba_classifier = get_simba_classifier(classifier_type=requested_classifier) if requested_classifier == "simba" else None
    classifier_mode = "adaptive_hmm"
    classifier_note = None
    if requested_classifier == "pytorch_temporal":
        if nn_classifier is not None:
            classifier_mode = "pytorch_temporal"
        else:
            classifier_note = (
                "PyTorch temporal classifier was requested, but no trained checkpoint was configured. "
                "Falling back to heuristic classification."
            )
    elif requested_classifier == "simba":
        if simba_classifier is not None:
            classifier_mode = "simba"
        else:
            classifier_note = (
                "SimBA classifier was requested, but no trained checkpoint was configured. "
                "Falling back to heuristic classification."
            )

    if stream_callback:
        stream_callback(
            {
                "type": "status",
                "pipeline_mode": initial_pipeline_mode,
                "classifier_mode": classifier_mode,
                "message": (
                    f"Classifier: {classifier_mode}. "
                    + ("Attempting DeepLabCut pose model path. " + pose_runtime_message
                       if pose_runtime_available
                       else "Using motion fallback path. " + pose_runtime_message)
                ),
            }
        )
        if classifier_note:
            stream_callback({"type": "status", "pipeline_mode": initial_pipeline_mode, "message": classifier_note})

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480

    raw_frame_count = 0
    frame_states: list[FrameState] = []
    centroids: list[tuple[float | None, float | None]] = []
    prev_gray = None
    prev_centroid = None
    prev_vector = None
    recent_speeds: list[float] = []
    motion_scores: list[float] = []
    direction_deltas: list[float] = []
    spatial_spans: list[float] = []
    bbox_area_ratios: list[float] = []
    pace_confined_scores: list[float] = []
    centroids_x: list[float | None] = []
    centroids_y: list[float | None] = []
    window_frames = max(5, int(fps * WINDOW_SECONDS))
    last_stream_label = "monitoring"

    # --- Phase 1: Extract motion features without holding decoded frames in RAM ---
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        timestamp_s = raw_frame_count / fps
        raw_frame_count += 1

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 0)

        motion_score = 0.0
        centroid_x = None
        centroid_y = None
        bbox_area_ratio = 0.0

        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            _, thresh = cv2.threshold(diff, 20, 255, cv2.THRESH_BINARY)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            thresh = cv2.dilate(thresh, None, iterations=2)
            motion_score = float(np.mean(thresh))

            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                contour = max(contours, key=cv2.contourArea)
                if cv2.contourArea(contour) > 120:
                    x, y, w, h = cv2.boundingRect(contour)
                    bbox_area_ratio = float((w * h) / max(width * height, 1))
                    moments = cv2.moments(contour)
                    if moments["m00"] != 0:
                        centroid_x = moments["m10"] / moments["m00"]
                        centroid_y = moments["m01"] / moments["m00"]

        prev_gray = gray

        if (centroid_x is None or centroid_y is None) and prev_centroid is not None:
            centroid_x, centroid_y = prev_centroid

        speed = 0.0
        direction_delta = 0.0
        if centroid_x is not None and centroid_y is not None and prev_centroid is not None:
            current_vector = np.array(
                [centroid_x - prev_centroid[0], centroid_y - prev_centroid[1]],
                dtype=float,
            )
            distance = float(np.hypot(current_vector[0], current_vector[1]))
            speed = distance * fps
            if prev_vector is not None and np.linalg.norm(current_vector) > 0 and np.linalg.norm(prev_vector) > 0:
                cosine = np.clip(
                    float(
                        np.dot(current_vector, prev_vector)
                        / (np.linalg.norm(current_vector) * np.linalg.norm(prev_vector))
                    ),
                    -1.0,
                    1.0,
                )
                direction_delta = float(np.degrees(np.arccos(cosine)))
            if np.linalg.norm(current_vector) > 0:
                prev_vector = current_vector
        if centroid_x is not None and centroid_y is not None:
            prev_centroid = (centroid_x, centroid_y)

        centroids.append((centroid_x, centroid_y))
        recent_speeds.append(speed)
        motion_scores.append(motion_score)
        direction_deltas.append(direction_delta)
        centroids_x.append(centroid_x)
        centroids_y.append(centroid_y)
        bbox_area_ratios.append(bbox_area_ratio)

        x_values = [x for x, _ in centroids[-window_frames:] if x is not None]
        y_values = [y for _, y in centroids[-window_frames:] if y is not None]
        spatial_span = (
            float(np.hypot(max(x_values) - min(x_values), max(y_values) - min(y_values)))
            if x_values and y_values
            else 0.0
        )
        spatial_spans.append(spatial_span)
        recent_avg_speed = float(np.mean(recent_speeds[-window_frames:])) if recent_speeds else 0.0
        pace_confined_score = recent_avg_speed / max(spatial_span, 1.0)
        pace_confined_scores.append(pace_confined_score)

        if stream_callback and raw_frame_count % max(5, int(fps // 2) or 1) == 0:
            stream_callback(
                {
                    "type": "progress",
                    "timestamp_s": float(timestamp_s),
                    "processed_frames": raw_frame_count,
                }
            )

    capture.release()

    if raw_frame_count == 0:
        raise RuntimeError("The uploaded video did not contain readable frames.")

    # --- Phase 2: Pose enrichment ---
    pose_enrichment = None
    pose_enrichment_error = None
    if pose_runtime_available:
        pose_enrichment, pose_enrichment_error = _maybe_pose_enrichment(input_path, fps)

    # --- Phase 3: Classify behavior (NN, SimBA, or adaptive HMM) ---
    def _append_states(labels: list[str], confidences: list[float]) -> None:
        nonlocal last_stream_label
        for i, (label, confidence) in enumerate(zip(labels, confidences)):
            if i >= raw_frame_count:
                break
            timestamp_s = i / fps
            frame_states.append(
                FrameState(
                    timestamp_s=float(timestamp_s),
                    label=label,
                    confidence=float(confidence),
                    speed=float(recent_speeds[i]),
                    motion_score=float(motion_scores[i]),
                    x=centroids_x[i],
                    y=centroids_y[i],
                    bbox_area_ratio=float(bbox_area_ratios[i]),
                    direction_delta=float(direction_deltas[i]),
                    spatial_span=float(spatial_spans[i]),
                    pace_confined_score=float(pace_confined_scores[i]),
                )
            )
            if (
                stream_callback
                and label != "monitoring"
                and label != last_stream_label
                and confidence >= 0.55
            ):
                stream_callback(
                    {
                        "type": "event",
                        "timestamp_s": float(timestamp_s),
                        "label": label,
                        "confidence": float(confidence),
                        "message": _note_for_label(label),
                    }
                )
                last_stream_label = label

    if classifier_mode == "simba":
        pose_df = pose_enrichment.get("pose_df") if pose_enrichment is not None else None
        if pose_df is None:
            classifier_note = (
                "SimBA classification requires DeepLabCut pose output, but pose enrichment was unavailable. "
                "Falling back to heuristic classification."
            )
            classifier_mode = "adaptive_hmm"
        else:
            classifications = simba_classifier.predict_from_dataframe(pose_df, fps=fps)
            _append_states(
                [cls["label"] for cls in classifications],
                [float(cls["confidence"]) for cls in classifications],
            )

    if classifier_mode == "pytorch_temporal":
        feature_matrix = extract_features_from_motion(
            speeds=recent_speeds,
            motion_scores=motion_scores,
            direction_deltas=direction_deltas,
            spatial_spans=spatial_spans,
            bbox_area_ratios=bbox_area_ratios,
            pace_confined_scores=pace_confined_scores,
            centroids_x=centroids_x,
            centroids_y=centroids_y,
            frame_width=width,
            frame_height=height,
            fps=fps,
            window_frames=window_frames,
        )
        feature_matrix = enrich_with_pose(feature_matrix, pose_enrichment, fps)
        classifications = nn_classifier.predict(feature_matrix)
        _append_states(
            [cls.label for cls in classifications],
            [float(cls.confidence) for cls in classifications],
        )

    if classifier_mode in {"heuristic", "adaptive_hmm"}:
        classifier_mode = "adaptive_hmm"
        feature_matrix = extract_features_from_motion(
            speeds=recent_speeds,
            motion_scores=motion_scores,
            direction_deltas=direction_deltas,
            spatial_spans=spatial_spans,
            bbox_area_ratios=bbox_area_ratios,
            pace_confined_scores=pace_confined_scores,
            centroids_x=centroids_x,
            centroids_y=centroids_y,
            frame_width=width,
            frame_height=height,
            fps=fps,
            window_frames=window_frames,
        )
        feature_matrix = enrich_with_pose(feature_matrix, pose_enrichment, fps)
        sequence = classify_feature_sequence(
            feature_matrix,
            frame_width=width,
            frame_height=height,
            fps=fps,
            condition=condition,
        )
        _append_states(sequence.labels, sequence.confidences)

    pipeline_mode = "hybrid_pose_motion" if pose_enrichment is not None else "motion_fallback"
    if classifier_mode == "pytorch_temporal":
        pipeline_mode = f"nn_{pipeline_mode}"
    elif classifier_mode == "simba":
        pipeline_mode = f"simba_{pipeline_mode}"
    elif classifier_mode == "adaptive_hmm":
        pipeline_mode = f"hmm_{pipeline_mode}"
    pose_summary = pose_enrichment["summary"] if pose_enrichment is not None else None

    label_counts = Counter(state.label for state in frame_states if state.label != "monitoring")
    dominant_behavior = label_counts.most_common(1)[0][0] if label_counts else "monitoring"
    bouts = _segment_bouts(frame_states)

    low_confidence_ratio = float(np.mean([state.confidence < 0.55 for state in frame_states])) if frame_states else 1.0
    base_priority = (
        "High" if low_confidence_ratio > 0.35 or dominant_behavior == "stereotypy_candidate"
        else "Medium" if low_confidence_ratio > 0.15
        else "Low"
    )

    behavior_share = {
        label: count / len(frame_states)
        for label, count in Counter(state.label for state in frame_states).items()
    }

    # Condition-aware review priority
    review_priority = condition_adjusted_review_priority(
        base_priority=base_priority,
        dominant_behavior=dominant_behavior,
        behavior_share=behavior_share,
        condition=condition,
    )

    # Condition anomaly detection
    condition_anomalies = detect_anomalies(behavior_share, condition)
    condition_report = generate_condition_report(behavior_share, condition, dominant_behavior)
    behavior_events = _generate_behavior_events(frame_states, bouts)
    analyst_messages = _generate_analyst_messages(
        behavior_events=behavior_events,
        dominant_behavior=dominant_behavior,
        pipeline_mode=pipeline_mode,
    )

    model_stack = [
        "deeplabcut_pose_estimation" if pose_enrichment is not None else "rat_motion_backbone",
        f"temporal_rat_behavior_classifier ({classifier_mode})",
        "agentic_event_interpreter",
    ]
    selected_models = _selected_models(pose_enrichment, classifier_mode)

    raw_annotated_path = artifact_dir / "annotated_raw.mp4"
    writer = cv2.VideoWriter(
        str(raw_annotated_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    replay = cv2.VideoCapture(str(input_path))
    frame_index = 0
    while True:
        ok, frame = replay.read()
        if not ok:
            break
        state = frame_states[frame_index] if frame_index < len(frame_states) else None
        if state is None:
            writer.write(frame)
        else:
            annotated_frame = _draw_overlay(frame, state, dominant_behavior)
            annotated_frame = _draw_pose_overlay(annotated_frame, pose_enrichment, frame_index)
            writer.write(annotated_frame)
        frame_index += 1
    replay.release()
    writer.release()

    annotated_path = artifact_dir / "annotated.mp4"
    try:
        _transcode_for_web(raw_annotated_path, annotated_path)
    except Exception:
        annotated_path = raw_annotated_path

    analysis_summary = _build_analysis_summary(
        dominant_behavior=dominant_behavior,
        behavior_share=behavior_share,
        bouts=bouts,
        review_priority=review_priority,
        pipeline_mode=pipeline_mode,
    )
    llm_summary = summarize_with_llm(
        dominant_behavior=dominant_behavior,
        review_priority=review_priority,
        bouts=bouts,
        behavior_share=behavior_share,
        pipeline_mode=pipeline_mode,
        condition=condition,
    )
    if llm_summary:
        analysis_summary = llm_summary

    timeline = {
        "analysis_id": analysis_id,
        "pipeline_mode": pipeline_mode,
        "model_stack": model_stack,
        "fps": float(fps),
        "frame_count": len(frame_states),
        "duration_s": float(len(frame_states) / fps),
        "dominant_behavior": dominant_behavior,
        "behavior_share": behavior_share,
        "bouts": bouts,
        "behavior_events": behavior_events,
        "analyst_messages": analyst_messages,
        "frames": [
            {
                "timestamp_s": state.timestamp_s,
                "label": state.label,
                "confidence": state.confidence,
                "speed": state.speed,
                "motion_score": state.motion_score,
            }
            for state in frame_states
        ],
        "analysis_summary": analysis_summary,
        "review_priority": review_priority,
        "classifier_mode": classifier_mode,
        "condition": condition,
        "condition_report": condition_report,
        "condition_anomalies": condition_anomalies,
        "notes": [
            "Adaptive HMM rat behavior classifier for drug-experiment screening on CPU.",
            f"Classifier mode: {classifier_mode}. Pipeline mode: {pipeline_mode}.",
            "Motion is normalized to this video's resolution and session percentiles, then decoded with Viterbi.",
            "When available, DeepLabCut pose outputs enrich the feature set for stronger classification.",
        ],
    }
    if pose_runtime_message:
        timeline["notes"].append(pose_runtime_message)
    if classifier_note:
        timeline["notes"].append(classifier_note)
    if pose_enrichment_error:
        timeline["notes"].append(f"DeepLabCut inference fallback reason: {pose_enrichment_error}")
    if pose_summary is not None:
        timeline["notes"].append(
            "DeepLabCut tracked: " + ", ".join(pose_summary.get("bodyparts_used", []))
        )
        missing = pose_summary.get("missing_priority_bodyparts", [])
        if missing:
            timeline["notes"].append(
                "Model coverage is still limited for richer rat behaviors; missing priority points include: "
                + ", ".join(missing[:8])
                + ("..." if len(missing) > 8 else "")
            )
    if condition_anomalies:
        for anomaly in condition_anomalies[:3]:
            timeline["notes"].append(
                f"Condition anomaly: {anomaly['behavior']} is {anomaly['direction'].replace('_', ' ')} "
                f"(observed {anomaly['observed']:.1%}, expected {anomaly['expected_range'][0]:.1%}-{anomaly['expected_range'][1]:.1%})"
            )
    if condition and condition_report:
        timeline["notes"].append(f"Condition: {condition_report['condition']} — {condition_report['condition_description']}")

    timeline_path = artifact_dir / "timeline.json"
    timeline_path.write_text(json.dumps(timeline, indent=2) + "\n", encoding="utf-8")

    return {
        "analysis_id": analysis_id,
        "pipeline_mode": pipeline_mode,
        "model_stack": model_stack,
        "classifier_mode": classifier_mode,
        "selected_models": selected_models,
        "source_video_path": str(input_path),
        "annotated_video_path": str(annotated_path),
        "timeline_path": str(timeline_path),
        "fps": float(fps),
        "frame_count": len(frame_states),
        "duration_s": float(len(frame_states) / fps),
        "dominant_behavior": dominant_behavior,
        "behavior_share": behavior_share,
        "bouts": bouts,
        "behavior_events": behavior_events,
        "analyst_messages": analyst_messages,
        "analysis_summary": analysis_summary,
        "review_priority": review_priority,
        "condition": condition,
        "condition_report": condition_report,
        "condition_anomalies": condition_anomalies,
        "notes": timeline["notes"],
    }


def annotate_behavior_video(
    video_path: str,
    output_root: str,
    condition: str | None = None,
    classifier_type: str | None = None,
) -> dict:
    return _run_behavior_analysis(
        video_path=video_path,
        output_root=output_root,
        condition=condition,
        classifier_type=classifier_type,
    )


def stream_behavior_analysis(
    video_path: str,
    output_root: str,
    condition: str | None = None,
    classifier_type: str | None = None,
):
    event_queue: "queue.Queue[dict]" = queue.Queue()

    def emit(event: dict) -> None:
        event_queue.put(event)

    def worker():
        try:
            result = _run_behavior_analysis(
                video_path=video_path,
                output_root=output_root,
                stream_callback=emit,
                condition=condition,
                classifier_type=classifier_type,
            )
            event_queue.put({"type": "result", "payload": result})
        except Exception as exc:
            event_queue.put({"type": "error", "detail": str(exc)})

    threading.Thread(target=worker, daemon=True).start()

    while True:
        event = event_queue.get()
        yield json.dumps(event) + "\n"
        if event.get("type") in {"result", "error"}:
            break
