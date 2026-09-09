import numpy as np

from app.adaptive_classifier import (
    classify_feature_sequence,
    emission_logits,
    stabilize_labels,
    viterbi,
    transition_log_matrix,
    OnlineAdaptiveClassifier,
)
from app.temporal_classifier import (
    LABEL_TO_IDX,
    NUM_FEATURES,
    extract_features_from_motion,
    heuristic_classify,
)


def _feature_matrix(width, height, speeds, spans, motion, directions):
    n = len(speeds)
    return extract_features_from_motion(
        speeds=list(speeds),
        motion_scores=list(motion),
        direction_deltas=list(directions),
        spatial_spans=list(spans),
        bbox_area_ratios=[0.02] * n,
        pace_confined_scores=[s / max(sp, 1.0) for s, sp in zip(speeds, spans)],
        centroids_x=[width * 0.4 + i for i in range(n)],
        centroids_y=[height * 0.5] * n,
        frame_width=width,
        frame_height=height,
        fps=20.0,
        window_frames=10,
    )


def test_adaptive_hmm_labels_travel_and_still_segments():
    n = 80
    speeds = np.concatenate([np.full(40, 90.0), np.zeros(40)])
    spans = np.concatenate([np.linspace(40, 180, 40), np.full(40, 8.0)])
    motion = np.concatenate([np.full(40, 40.0), np.full(40, 0.4)])
    directions = np.zeros(n)
    features = _feature_matrix(240, 180, speeds, spans, motion, directions)

    result = classify_feature_sequence(features, frame_width=240, frame_height=180, fps=20.0)
    travel = set(result.labels[:40])
    still = set(result.labels[40:])
    assert travel & {"locomotor_burst", "exploration"}
    assert still & {"freezing_candidate", "resting"}


def test_viterbi_removes_single_frame_flicker():
    labels_flicker = (
        ["exploration"] * 12
        + ["freezing_candidate"]
        + ["exploration"] * 12
    )
    smoothed = stabilize_labels(labels_flicker, min_run=4)
    assert smoothed.count("freezing_candidate") == 0
    assert set(smoothed) == {"exploration"}


def test_scale_invariance_across_resolutions():
    n = 60
    small_w, small_h = 240, 180
    large_w, large_h = 960, 720
    scale = large_w / small_w

    small_speed = np.full(n, 70.0)
    small_span = np.linspace(50, 140, n)
    small_motion = np.full(n, 30.0)
    small_dir = np.zeros(n)
    small_feat = _feature_matrix(small_w, small_h, small_speed, small_span, small_motion, small_dir)
    large_feat = _feature_matrix(
        large_w,
        large_h,
        small_speed * scale,
        small_span * scale,
        small_motion,
        small_dir,
    )

    small_result = classify_feature_sequence(small_feat, frame_width=small_w, frame_height=small_h, fps=20.0)
    large_result = classify_feature_sequence(large_feat, frame_width=large_w, frame_height=large_h, fps=20.0)
    assert small_result.labels[n // 2] in {"exploration", "locomotor_burst"}
    assert large_result.labels[n // 2] in {"exploration", "locomotor_burst"}
    agree = sum(a == b for a, b in zip(small_result.labels, large_result.labels)) / n
    assert agree >= 0.7


def test_meth_prior_boosts_stereotypy_on_confined_motion():
    n = 90
    speeds = np.full(n, 55.0)
    spans = np.full(n, 28.0)
    motion = np.full(n, 35.0)
    directions = np.full(n, 110.0)
    features = _feature_matrix(320, 240, speeds, spans, motion, directions)

    baseline = classify_feature_sequence(features, frame_width=320, frame_height=240, fps=20.0, condition="baseline")
    meth = classify_feature_sequence(features, frame_width=320, frame_height=240, fps=20.0, condition="meth")
    meth_share = meth.labels.count("stereotypy_candidate") / n
    base_share = baseline.labels.count("stereotypy_candidate") / n
    assert meth_share >= base_share
    assert meth_share > 0.2


def test_online_classifier_tracks_state_changes():
    clf = OnlineAdaptiveClassifier(frame_width=320, frame_height=240, fps=20.0)
    labels = []
    for i in range(40):
        label, conf = clf.update(
            speed=80.0,
            motion_score=40.0,
            direction_delta=8.0,
            spatial_span=40 + i * 4,
            pace_confined_score=0.4,
        )
        labels.append(label)
        assert 0.0 < conf <= 1.0
    assert any(label in {"exploration", "locomotor_burst"} for label in labels[-10:])


def test_legacy_heuristic_still_available_for_weak_labels():
    label, conf = heuristic_classify(
        speed=80,
        motion_score=12,
        direction_delta=10,
        spatial_span=120,
        pace_confined_score=0.4,
        recent_avg_speed=70,
        recent_direction_mean=12,
        width=320,
    )
    assert label == "locomotor_burst"
    assert conf > 0.5


def test_viterbi_path_length_matches_emissions():
    features = np.zeros((25, NUM_FEATURES), dtype=np.float32)
    features[:, 0] = 8.0
    logits, _ = emission_logits(features, 240, 180, condition=None)
    path, conf = viterbi(logits, transition_log_matrix(None))
    assert len(path) == 25
    assert len(conf) == 25
    assert set(path).issubset(set(LABEL_TO_IDX.values()))
