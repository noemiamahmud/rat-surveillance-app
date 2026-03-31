# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Rat Behavior Intelligence — a browser-based rat pose and behavior analysis app. Users upload rat videos via a web frontend, the FastAPI backend runs a staged behavior classification pipeline (motion features → optional DeepLabCut pose enrichment → optional LLM summary), and returns an annotated video, timestamped behavior events, and an AI observer chat feed.

This is a heuristic-first system, not a trained classifier. It's best used as a screening/review tool.

## Commands

```bash
# Run backend (serves frontend too)
cd backend
source .venv/bin/activate
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
# App at http://127.0.0.1:8000, Swagger at http://127.0.0.1:8000/docs

# Run tests
cd backend
pytest tests/test_api.py

# Run a single test
pytest tests/test_api.py::test_function_name -v
```

No frontend build step — static HTML/CSS/JS served directly by FastAPI.

## Architecture

**Backend** (`backend/app/`): FastAPI (Python 3.13)
- `main.py` — App init, CORS, static file mounts, route inclusion
- `routes.py` — REST endpoints: video upload (blocking + streaming NDJSON), DLC-based analysis, camera control, session management
- `video_annotation.py` — Core pipeline: frame-by-frame motion feature extraction → behavior classification (8 labels) → bout segmentation → annotated MP4 generation → optional LLM summary
- `pose_analysis.py` — DeepLabCut CSV processing, bodypart normalization, centroid tracking, pose-derived metrics
- `dlc_runner.py` — Orchestrates DeepLabCut inference on videos, handles PyTorch/TF engine detection
- `observer_llm.py` — Optional LLM summarization via OpenAI API or local llama.cpp server
- `config.py` — Pydantic settings from `.env`, data directory paths, LLM provider config
- `schemas.py` — Request/response Pydantic models

**Frontend** (`frontend/`): Vanilla HTML5/CSS3/ES6 JS (no framework, no build)
- `app.js` — Upload handling, NDJSON streaming parser, result rendering
- `index.html` — Single-page UI with drag-drop upload, video player, results display

**DeepLabCut** (`dlc_project/`): PyTorch ResNet-50 pose model (5-point skeleton: nose, upper_back, lower_back, tail_base, tail_tip). Expanded 16-point template exists but isn't active yet.

**Local LLM** (`models/`): Qwen2.5-7B GGUF for llama.cpp — optional, not auto-launched.

## Key Design Patterns

- **Graceful degradation**: Pipeline runs without DLC (motion-only fallback), without LLM (rule-based messages), and without FFmpeg (raw MP4).
- **Dual upload endpoints**: `/api/analysis/upload-video` (blocking) and `/api/analysis/upload-video-stream` (streaming NDJSON progress events).
- **Behavior classification**: 8 heuristic labels based on speed, direction changes, motion score, and spatial span. Bouts require minimum 0.6s duration.
- **Artifact-per-analysis**: Each analysis gets a unique ID; results stored in `backend/data/processed/annotated_sessions/{analysis_id}/`.
- **No database**: All state is filesystem-based (JSON files, video artifacts).

## Environment Configuration

Copy `backend/.env.example` to `backend/.env`. Key variables:
- `LLM_PROVIDER` — `openai`, `llama_cpp`, or `none`
- `LLM_BASE_URL` — API endpoint for LLM provider
- `OPENAI_API_KEY` — Required if using OpenAI
- `LLAMA_CPP_MODEL` — Model name for local llama.cpp server


## Project Goal

This project is a neural-network-based behavior classification system for rats undergoing drug experiments (meth, alcohol, etc.). The pipeline follows the DLC → SimBA methodology: (1) capture video of operant self-administration sessions, (2) extract pose estimation with DeepLabCut, (3) generate behavioral classifiers with SimBA or a custom PyTorch/TensorFlow temporal model, (4) compare automated classification against standard Med-PC lever response data.

## TODO — Rebuild Roadmap

### Phase 1: Expand Pose Tracking (DLC)
- Retrain DeepLabCut with the 16-point expanded skeleton (`dlc_project/rat_pose_expanded_config_template.yaml`) — current 5-point model (nose, upper_back, lower_back, tail_base, tail_tip) is insufficient for grooming, paw interaction, and ear-based cues
- Label 200-400 new frames across meth, alcohol, and baseline conditions for training diversity
- The expanded skeleton adds: ears, neck, shoulder, spine_mid, hip, tail_mid, and all 4 paws
- `pose_analysis.py` already handles richer skeletons (body_curvature, grooming_motion, turning_rate compute automatically from extra keypoints) — no code changes needed there

### Phase 2: Replace Heuristic Classifier with Neural Network
- Current behavior classification is entirely heuristic thresholds in `video_annotation.py` lines 435-463 (e.g., speed > 70 → locomotor_burst, speed < 5 → freezing_candidate)
- Replace with a trained temporal classifier. Two viable architectures:
  - **SimBA integration**: LSTM-based classifier that reads DLC pose CSVs directly, pre-trained on rodent behavior datasets, fine-tunable on drug-specific labels
  - **Custom PyTorch model**: Sequence model (LSTM/Transformer) on pose features + motion features, or 3D CNN (SlowFast/ResNet3D) on short video clips with pose heatmap overlay
- Keep the `FrameState` dataclass interface — NN output populates same fields (label, confidence) so bout segmentation, video annotation, and streaming all remain unchanged
- Add `CLASSIFIER_TYPE` config option: `heuristic` (current fallback), `simba`, or `pytorch_temporal`

### Phase 3: Drug Condition-Aware Analysis
- Schemas already support `condition: Optional[str]` but it's ignored during classification
- Make condition influence analysis: inject into LLM prompts, flag condition-specific anomalies (e.g., stereotypy >> baseline in meth = expected; in baseline = abnormal)
- Add normative behavior distributions per condition for comparison reporting
- Stratify bout summaries by condition in the response

### Phase 4: SimBA Integration
- SimBA reads DLC CSV output directly — fits between `dlc_runner.py` and `video_annotation.py`
- Pipeline: Video → DLC pose extraction → SimBA feature engineering → SimBA classifier → frame-level predictions
- SimBA provides: temporal LSTM modeling, feature importance scores, region-of-interest quadrant time analysis (comparable to Med-PC lever data)
- Requires expanded skeleton (Phase 1) — SimBA needs ≥10 keypoints for reliable results
- Add `simba` to requirements.txt, new `simba_classifier.py` module, new `/api/analysis/simba-classify` endpoint

### Phase 5: Training Data & Validation
- Collect and label 50-100 videos (15-30min each) across drug conditions with frame-level behavior annotations
- Use current heuristic output as weak supervision seed, then human-correct boundary cases (grooming vs. freezing, stereotypy vs. turning)
- Cross-validate classifier against hand-labeled ground truth
- Compare automated quadrant time data against Med-PC lever response data

### Immediate Fixes
- Fix LLM API call in `observer_llm.py` — uses `client.responses.create()` which is non-standard; should be `client.chat.completions.create()`
- `behavior.py` is a stub (simple frame differencing, not used) — delete or replace
- `video_stream.py` session persistence raises `NotImplementedError` — implement or remove
- Add `torch`, `torchvision`, and `deeplabcut` to requirements.txt (currently expected as separate installs)

## Current Gaps Summary

| Area | Now | Target |
|------|-----|--------|
| Behavior classification | Heuristic thresholds | Trained NN (SimBA or PyTorch temporal) |
| Pose skeleton | 5 keypoints | 16 keypoints (template ready) |
| Drug condition | Stored in metadata, unused | Drives analysis thresholds and reporting |
| Temporal modeling | Frame-by-frame | LSTM/Transformer sequence modeling |
| Validation | None | Cross-validated against labeled data + Med-PC |
| Session persistence | NotImplementedError | Filesystem or SQLite |
