from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.temporal_classifier import FEATURE_NAMES, NUM_FEATURES
from app.training import (
    apply_review,
    export_dataset,
    export_review_queue,
    split_train_val,
    write_training_frames_csv,
)


@dataclass
class _State:
    timestamp_s: float
    label: str
    confidence: float


def _session_dir(tmp_path: Path, analysis_id: str, labels: list[str], confidences: list[float] | None = None):
    folder = tmp_path / analysis_id
    folder.mkdir()
    n = len(labels)
    features = np.zeros((n, NUM_FEATURES), dtype=np.float32)
    features[:, 0] = np.linspace(0, 80, n)
    states = [
        _State(timestamp_s=i / 10.0, label=labels[i], confidence=(confidences or [0.8] * n)[i])
        for i in range(n)
    ]
    write_training_frames_csv(
        folder / "training_frames.csv",
        feature_matrix=features,
        frame_states=states,
        analysis_id=analysis_id,
        condition="baseline",
        fps=10.0,
        label_source="adaptive_hmm",
    )
    bouts = []
    start = 0
    for i in range(1, n + 1):
        if i == n or labels[i] != labels[start]:
            bouts.append(
                {
                    "label": labels[start],
                    "start_s": start / 10.0,
                    "end_s": (i - 1) / 10.0,
                    "duration_s": max((i - 1 - start) / 10.0, 0.0),
                    "confidence": float(np.mean((confidences or [0.8] * n)[start:i])),
                }
            )
            start = i
    (folder / "timeline.json").write_text(
        __import__("json").dumps({"analysis_id": analysis_id, "condition": "baseline", "bouts": bouts}),
        encoding="utf-8",
    )
    return folder


def test_export_dataset_and_split(tmp_path):
    _session_dir(tmp_path, "aaa", ["exploration"] * 20 + ["resting"] * 20)
    _session_dir(tmp_path, "bbb", ["locomotor_burst"] * 20 + ["hesitation"] * 20)
    out = tmp_path / "dataset.csv"
    result = export_dataset(out, sessions_dir=tmp_path)
    assert result["sessions"] == 2
    assert result["frames"] == 80
    df = pd.read_csv(out)
    assert list(FEATURE_NAMES) == [col for col in FEATURE_NAMES if col in df.columns]
    train_x, train_y, val_x, val_y = split_train_val(str(out), val_fraction=0.5)
    assert len(train_x) + len(val_x) == 80
    assert train_x.shape[1] == NUM_FEATURES
    assert len(train_y) > 0 and len(val_y) > 0


def test_review_and_apply(tmp_path):
    labels = ["grooming_candidate"] * 15 + ["exploration"] * 15
    conf = [0.4] * 15 + [0.9] * 15
    _session_dir(tmp_path, "sess1", labels, conf)
    review_path = tmp_path / "review.csv"
    exported = export_review_queue(review_path, sessions_dir=tmp_path, min_confidence=0.6, min_bout_s=0.3)
    assert exported["bouts"] >= 1
    review = pd.read_csv(review_path, dtype={"corrected_label": str})
    review["corrected_label"] = review["corrected_label"].fillna("")
    review.loc[0, "corrected_label"] = "freezing_candidate"
    review.to_csv(review_path, index=False)
    applied = apply_review(review_path, sessions_dir=tmp_path)
    assert applied["patched_frames"] > 0
    frames = pd.read_csv(tmp_path / "sess1" / "training_frames.csv")
    assert "freezing_candidate" in set(frames["label"])
    assert (frames["label_source"] == "human").any()


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("torch") is None,
    reason="PyTorch is required to train the temporal classifier.",
)
def test_train_from_csv_on_tiny_set(tmp_path):
    from app.training import train_from_csv

    labels = (["exploration"] * 40) + (["resting"] * 40)
    _session_dir(tmp_path, "trainA", labels)
    _session_dir(tmp_path, "trainB", (["locomotor_burst"] * 40) + (["hesitation"] * 40))
    dataset = tmp_path / "dataset.csv"
    export_dataset(dataset, sessions_dir=tmp_path)
    result = train_from_csv(
        str(dataset),
        model_type="lstm",
        epochs=2,
        batch_size=4,
        output_dir=str(tmp_path / "model"),
    )
    assert Path(result["model_path"]).exists()
    assert (tmp_path / "model" / "feature_scaler.pkl").exists()
