import pandas as pd
import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.main import app
from app.pose_analysis import summarize_pose_analysis


def _write_synthetic_dlc_csv(csv_path):
    frames = 120
    scorer = "scorer"
    columns = pd.MultiIndex.from_tuples(
        [
            (scorer, "nose", "x"),
            (scorer, "nose", "y"),
            (scorer, "nose", "likelihood"),
            (scorer, "tailbase", "x"),
            (scorer, "tailbase", "y"),
            (scorer, "tailbase", "likelihood"),
        ]
    )

    rows = []
    for frame in range(frames):
        if frame < 40:
            x = 100 + frame * 3
            y = 120
        elif frame < 80:
            x = 220 + ((frame % 2) * 40)
            y = 120
        else:
            x = 225
            y = 122

        rows.append([x, y, 0.99, x - 18, y + 10, 0.98])

    pd.DataFrame(rows, columns=columns).to_csv(csv_path, index=False)


def test_offline_dlc_analysis_endpoint(tmp_path):
    csv_path = tmp_path / "synthetic_dlc.csv"
    _write_synthetic_dlc_csv(csv_path)

    client = TestClient(app)
    response = client.post(
        "/api/analysis/from-dlc",
        json={
            "dlc_csv_path": str(csv_path),
            "fps": 30,
            "subject_id": "rat-a",
            "condition": "amphetamine",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["subject_id"] == "rat-a"
    assert payload["condition"] == "amphetamine"
    assert "nose" in payload["bodyparts_used"]
    assert payload["metrics"]["valid_pose_ratio"] > 0.95
    assert payload["metrics"]["distance_px"] > 0
    assert payload["metrics"]["pace_confined_score"] > 1
    assert any(
        bout["label"] in {"exploratory_locomotion", "immobility_bout"}
        for bout in payload["bouts"]
    )


def test_upload_video_analysis_endpoint(tmp_path):
    video_path = tmp_path / "synthetic.mp4"
    fps = 10
    width, height = 240, 180
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    x = 40
    for frame_idx in range(50):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        if frame_idx < 20:
            x += 6
        elif frame_idx < 35:
            x += 1
        else:
            x = 150 + (10 if frame_idx % 2 == 0 else -10)
        cv2.circle(frame, (max(20, min(width - 20, x)), 90), 16, (255, 255, 255), -1)
        writer.write(frame)
    writer.release()

    client = TestClient(app)
    with video_path.open("rb") as handle:
        response = client.post(
            "/api/analysis/upload-video",
            files={"file": ("synthetic.mp4", handle, "video/mp4")},
            data={"condition": "baseline", "classifier_type": "pytorch_temporal"},
        )

    assert response.status_code == 200
    payload = response.json()
    assert payload["annotated_video_url"].startswith("/artifacts/")
    assert payload["source_video_url"].startswith("/uploads/")
    assert "analysis_summary" in payload
    assert payload["review_priority"] in {"Low", "Medium", "High"}
    assert payload["pipeline_mode"] in {"motion_fallback", "hybrid_pose_motion"}
    assert payload["classifier_mode"] == "heuristic"
    assert payload["condition"] == "baseline"
    assert payload["condition_report"]["condition"] == "Baseline"
    assert isinstance(payload["condition_anomalies"], list)
    assert isinstance(payload["model_stack"], list) and payload["model_stack"]
    assert isinstance(payload["selected_models"], dict)
    assert "pose_model" in payload["selected_models"]
    assert isinstance(payload["behavior_events"], list)
    assert isinstance(payload["analyst_messages"], list) and payload["analyst_messages"]
    assert payload["frame_count"] > 0
    assert payload["duration_s"] > 0
    assert isinstance(payload["bouts"], list)
    assert any("falling back to heuristic classification" in note.lower() for note in payload["notes"])


def test_pose_analysis_supports_richer_rat_body_schema():
    frames = 60
    scorer = "scorer"
    columns = pd.MultiIndex.from_tuples(
        [
            (scorer, "nose", "x"),
            (scorer, "nose", "y"),
            (scorer, "nose", "likelihood"),
            (scorer, "left_ear", "x"),
            (scorer, "left_ear", "y"),
            (scorer, "left_ear", "likelihood"),
            (scorer, "right_ear", "x"),
            (scorer, "right_ear", "y"),
            (scorer, "right_ear", "likelihood"),
            (scorer, "upper_back", "x"),
            (scorer, "upper_back", "y"),
            (scorer, "upper_back", "likelihood"),
            (scorer, "lower_back", "x"),
            (scorer, "lower_back", "y"),
            (scorer, "lower_back", "likelihood"),
            (scorer, "tail_base", "x"),
            (scorer, "tail_base", "y"),
            (scorer, "tail_base", "likelihood"),
            (scorer, "tail_mid", "x"),
            (scorer, "tail_mid", "y"),
            (scorer, "tail_mid", "likelihood"),
            (scorer, "tail_tip", "x"),
            (scorer, "tail_tip", "y"),
            (scorer, "tail_tip", "likelihood"),
            (scorer, "left_forepaw", "x"),
            (scorer, "left_forepaw", "y"),
            (scorer, "left_forepaw", "likelihood"),
            (scorer, "right_forepaw", "x"),
            (scorer, "right_forepaw", "y"),
            (scorer, "right_forepaw", "likelihood"),
        ]
    )

    rows = []
    for frame in range(frames):
        base_x = 120 + frame * 1.5
        base_y = 110 + (2 if frame % 10 < 5 else -2)
        paw_offset = 7 if frame % 6 < 3 else -7
        rows.append(
            [
                base_x + 14, base_y - 10, 0.99,
                base_x + 6, base_y - 14, 0.98,
                base_x + 10, base_y - 14, 0.98,
                base_x + 4, base_y - 4, 0.99,
                base_x - 4, base_y + 2, 0.99,
                base_x - 12, base_y + 8, 0.99,
                base_x - 22, base_y + 10, 0.98,
                base_x - 34, base_y + 11, 0.98,
                base_x + 2, base_y + paw_offset, 0.96,
                base_x + 8, base_y - paw_offset, 0.96,
            ]
        )

    df = pd.DataFrame(rows, columns=columns)
    analysis = summarize_pose_analysis(df, fps=30)

    assert "left_ear" in analysis["bodyparts_used"]
    assert "left_forepaw" in analysis["bodyparts_used"]
    assert ("tail_base", "tail_mid") in analysis["skeleton_pairs"]
    assert analysis["metrics"]["tracked_bodyparts_count"] >= 8
    assert analysis["metrics"]["mean_turning_rate_deg_s"] >= 0
    assert "The rat pose analyzer can exploit a richer body schema" in analysis["notes"][2]
