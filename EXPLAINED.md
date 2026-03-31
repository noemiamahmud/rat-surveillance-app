# Rat Surveillance App Explained

## What this application is

This repository is a browser-based rat behavior analysis system for preclinical research workflows. The current app lets a user upload a rat video in the browser, runs a backend analysis pipeline over the clip, and returns:

- an annotated output video
- frame-level and bout-level behavior results
- a short summary of the session
- timestamped observer-style messages
- condition-aware review notes when an experimental condition is provided

The project is aimed at drug-study and behavioral-screening use cases, especially sessions where a researcher wants a faster first-pass review of locomotion, hesitation, freezing-like pauses, grooming-like activity, stereotypy candidates, and turning patterns.

The codebase already contains infrastructure for a stronger pose-first and neural-network-first pipeline, but the default production path is still a heuristic-first classifier with optional upgrades when local models and dependencies are available.

## What it aims to achieve

The long-term goal is to turn rodent video into structured, reviewable behavioral data that can support experimental analysis in areas like:

- stimulant exposure such as methamphetamine, amphetamine, and cocaine
- depressant or sedative conditions such as alcohol
- comparative baseline or saline control sessions
- eventual comparison against downstream lab measures such as operant chamber or Med-PC style outputs

In practical terms, the app is trying to become a staged pipeline like:

1. collect or upload rat video
2. estimate pose with DeepLabCut
3. classify behavior over time with a temporal model or SimBA-style classifier
4. surface abnormal or condition-relevant segments for human review
5. export artifacts a researcher can inspect without replaying the full raw video manually

Today, the repository is between steps 2 and 3. It can already run full uploads end to end, but it still falls back to hand-tuned motion rules whenever trained behavior checkpoints or DLC runtime support are missing.

## How the application currently works

### High-level architecture

The repo has three primary layers:

- `frontend/`: a static single-page interface in plain HTML, CSS, and JavaScript
- `backend/app/`: a FastAPI backend that also serves the frontend and generated artifacts
- `dlc_project/`: a bundled DeepLabCut project, training data, and project config used for pose estimation and labeling workflows

There is also a local-model area under `models/` containing a GGUF model intended for optional local LLM summarization via a separate `llama.cpp` server.

## Main user flow

The main implemented path is the upload flow:

1. The user opens the frontend served by FastAPI.
2. The user selects a video, optionally selects an experimental condition, optionally requests a classifier mode, and can keep live observer updates enabled.
3. The frontend posts the file to either:
   - `/api/analysis/upload-video` for a blocking request, or
   - `/api/analysis/upload-video-stream` for streaming NDJSON progress updates.
4. The backend stores the uploaded file in `backend/data/uploads/`.
5. The backend runs `_run_behavior_analysis()` in [`backend/app/video_annotation.py`](/Users/noemiamahmud/rat-surveillance-app/backend/app/video_annotation.py).
6. The backend extracts motion features frame by frame from the raw video using OpenCV.
7. If DeepLabCut is usable in the active environment, the backend also runs pose estimation and pose summarization.
8. The backend chooses a classifier mode:
   - heuristic fallback
   - PyTorch temporal classifier if a trained checkpoint is configured
   - SimBA-style classifier if configured and a trained checkpoint is present
9. The backend segments bouts, creates event messages, computes behavior share, review priority, and condition anomalies.
10. The backend renders an annotated MP4 with overlays and writes a `timeline.json`.
11. The frontend displays the annotated video, summary cards, behavior chips, notes, observer feed, and detected bouts.

## What the backend actually computes today

### 1. Motion-first feature extraction

The analysis starts with plain video processing:

- frame differencing between consecutive grayscale frames
- thresholding and morphology to isolate movement
- contour extraction to estimate a main moving blob
- centroid tracking
- speed
- direction change
- spatial span over a rolling window
- bounding-box area ratio
- a pace-versus-confinement score

This is the baseline path that always works if the video can be opened, even when pose estimation is unavailable.

### 2. Optional DeepLabCut pose enrichment

If the configured DLC project exists and the runtime imports cleanly, the backend attempts to:

- run DLC inference on the uploaded video
- read the generated CSV
- normalize bodypart names
- compute pose-derived metrics such as turning rate, body length, tail extension, body curvature, and grooming-related local motion
- attach per-frame keypoints for overlay rendering

The current code is built to support a rich 16-point rat skeleton and the bundled DLC project config lists 16 bodyparts. Some older project notes still refer to an earlier compact pose setup, so the safest interpretation is that the repository is prepared for the richer schema, but effective runtime behavior still depends on the installed DLC environment and whichever trained snapshot is actually active. The app treats DLC as optional and degrades to motion-only classification when runtime support is missing or inference fails.

### 3. Behavior classification

There are three classification modes in the codebase.

#### Heuristic mode

This is the default fallback and the most reliable current path. It uses threshold rules on motion features to label frames as:

- `exploration`
- `locomotor_burst`
- `freezing_candidate`
- `grooming_candidate`
- `hesitation`
- `stereotypy_candidate`
- `turning_pattern`
- `resting`
- `monitoring`

This logic lives in [`backend/app/temporal_classifier.py`](/Users/noemiamahmud/rat-surveillance-app/backend/app/temporal_classifier.py) and is what most runs will use unless trained checkpoints are configured.

#### PyTorch temporal classifier mode

The repo contains:

- LSTM and Transformer model definitions
- feature extraction utilities
- model loading
- training code

But this mode only becomes active if:

- PyTorch is installed
- `CLASSIFIER_TYPE=pytorch_temporal`
- `CLASSIFIER_MODEL_PATH` points to an existing trained checkpoint

If those conditions are not met, the code explicitly falls back to heuristics.

#### SimBA mode

The repo also contains a SimBA-inspired classifier path:

- DLC-pose feature engineering
- an LSTM-based classifier wrapper
- frame-level predictions
- optional quadrant-time analysis

As with the PyTorch temporal path, this is only active when a trained checkpoint is configured. Otherwise the main analysis route falls back to heuristics.

### 4. Bouts, summaries, and artifacts

After frame labels are generated, the backend:

- groups continuous labels into bouts
- ignores short or low-signal segments
- generates timestamped behavior events
- generates observer-style messages
- computes a dominant behavior and behavior-share breakdown
- writes a timeline JSON artifact
- draws labels and metrics onto every video frame
- transcodes the annotated output to a browser-friendly MP4 with `ffmpeg` when available

Artifacts are written per analysis under:

- `backend/data/uploads/`
- `backend/data/processed/annotated_sessions/<analysis_id>/`

## Condition-aware analysis

The app already has a condition profile layer in [`backend/app/condition_profiles.py`](/Users/noemiamahmud/rat-surveillance-app/backend/app/condition_profiles.py). When the user selects a condition like `baseline`, `meth`, `amphetamine`, `cocaine`, `alcohol`, or `morphine`, the backend can:

- compare observed behavior distribution to expected ranges
- detect anomalies such as behavior being above or below the expected range
- elevate review priority for suspicious or condition-relevant patterns
- add a structured condition report to the response

This is not yet a learned condition-aware classifier. It is a rules-and-reference-distribution layer on top of the analysis output.

## Observer summary layer

The UI presents an AI observer feed, but there are two different mechanisms behind it:

- live event-style observer messages are rule-based and generated directly by the backend
- the final natural-language summary can optionally be upgraded by an LLM

The LLM path is optional. The code supports:

- OpenAI via the Python SDK
- a local `llama.cpp` OpenAI-compatible server

If no provider is configured, the app still works and falls back to a generated heuristic summary.

## Frontend behavior

The frontend is intentionally simple:

- no framework
- no bundler
- no build step
- just `index.html`, `styles.css`, and `app.js`

Its responsibilities are:

- file selection
- condition and classifier selection
- toggling live observer updates
- posting multipart uploads
- parsing NDJSON progress events for streaming mode
- rendering returned artifacts and metadata

Because the backend mounts the frontend directory directly, starting the FastAPI app is enough to run the whole application locally.

## Secondary and partial features

Several features exist but are not the main maturity path yet:

### Live camera and session recording

`routes.py` and [`backend/app/video_stream.py`](/Users/noemiamahmud/rat-surveillance-app/backend/app/video_stream.py) support:

- RTSP camera connection
- MJPEG live feed
- starting and stopping recording sessions
- saving a simple behavior timeline per session

This path uses [`backend/app/behavior.py`](/Users/noemiamahmud/rat-surveillance-app/backend/app/behavior.py), which is a very lightweight frame-differencing stub that only emits `rat_active` and `rat_idle`. It is not the same as the richer upload-analysis pipeline.

### DLC labeling workflow

The repository also includes:

- a DeepLabCut project in `dlc_project/`
- a Python labeling helper in [`backend/app/dlc_labeling.py`](/Users/noemiamahmud/rat-surveillance-app/backend/app/dlc_labeling.py)
- a shell wrapper in [`backend/run_dlc_labeling.sh`](/Users/noemiamahmud/rat-surveillance-app/backend/run_dlc_labeling.sh)

This part of the repo exists to support collecting more labeled rat pose data and retraining the pose model.

### Temporal-model training pipeline

[`backend/app/training.py`](/Users/noemiamahmud/rat-surveillance-app/backend/app/training.py) contains training utilities for future learned behavior classification:

- loading labeled datasets
- feature normalization
- class weighting
- LSTM or Transformer training
- checkpoint saving
- metadata saving
- cross-validation support

This makes it clear the project intends to move away from heuristics, but the training workflow is still infrastructure rather than the current default product behavior.

## Tech stack

### Backend

- Python
- FastAPI
- Uvicorn
- Pydantic
- OpenCV
- NumPy
- pandas
- optional PyTorch
- optional scikit-learn
- optional OpenAI SDK
- optional DeepLabCut
- optional `ffmpeg`

### Frontend

- HTML5
- CSS3
- vanilla JavaScript
- browser video playback
- `fetch()` plus streaming response parsing

### ML / analysis stack

- DeepLabCut for pose estimation
- custom heuristic behavior classification
- optional PyTorch temporal classifier
- optional SimBA-style classifier path
- optional LLM summary layer

### Data and storage model

- filesystem-based storage only
- uploaded videos saved directly on disk
- processed artifacts saved per analysis directory
- no database
- no message queue
- no background job runner

## Repository layout

- `frontend/`: static UI
- `backend/app/main.py`: app bootstrap, CORS, route inclusion, static mounting
- `backend/app/routes.py`: API surface
- `backend/app/video_annotation.py`: main analysis pipeline
- `backend/app/temporal_classifier.py`: heuristic logic plus temporal-model loading
- `backend/app/pose_analysis.py`: DLC CSV analysis and pose metrics
- `backend/app/dlc_runner.py`: DLC inference orchestration
- `backend/app/simba_classifier.py`: SimBA-inspired classifier wrapper
- `backend/app/condition_profiles.py`: condition-aware anomaly logic
- `backend/app/observer_llm.py`: optional LLM summary integration
- `backend/app/video_stream.py`: RTSP/live session support
- `backend/app/training.py`: model-training utilities
- `backend/tests/test_api.py`: API and analysis tests
- `dlc_project/`: DLC project, labels, config, and training assets
- `models/`: local LLM artifact storage

## What is already strong in this codebase

- The upload-analysis UX works end to end without a frontend build system.
- The backend degrades gracefully when optional ML dependencies are missing.
- Artifact generation is clear and filesystem-based, which makes debugging easy.
- The pose analysis code is already designed for a richer rat skeleton, not just a minimal spine-tail chain.
- The repo already includes infrastructure for future learned models rather than requiring a rewrite later.

## Current limitations

- The main production classification path is still heuristic.
- Neural behavior classifiers require checkpoints that are not configured by default.
- DeepLabCut inference depends on a compatible local DLC runtime.
- Live camera analysis is much simpler than offline upload analysis.
- There is no database, async job queue, or persistent deployment-oriented workflow.
- Long analyses still run inline rather than as background tasks.
- Condition awareness is rule-based rather than learned from labeled data.

## Future goals implied by the codebase

The future direction is fairly clear from the implemented modules and project comments.

### 1. Make pose extraction consistently available

The first major goal is for uploaded videos to reliably pass through DeepLabCut instead of often dropping into motion fallback. That means:

- maintaining a working DLC runtime
- preserving compatibility with the bundled project config
- continuing to improve labeling and retraining workflows

### 2. Move from heuristics to trained temporal behavior models

The repo already contains the training and inference scaffolding for this. The likely intended evolution is:

- keep heuristics as a fallback
- train sequence models on labeled behavior data
- swap those predictions into the existing frame-state and bout pipeline

### 3. Expand and validate the 16-point rat skeleton workflow

The code is already built around a richer body schema including:

- ears
- neck and torso landmarks
- paws
- more tail points

This matters because grooming, turning, hesitation, posture shifts, and stereotypy are much easier to separate when pose detail is better than simple centroid motion.

### 4. Improve condition-specific analysis

The project already stores and uses condition names, but future work likely includes:

- condition-aware thresholds
- learned condition-specific priors
- stronger anomaly reporting
- better comparison to control distributions

### 5. Support stronger research outputs

The SimBA path and quadrant-time code suggest future goals such as:

- operant-chamber region analysis
- alignment with Med-PC or related behavioral data
- better experimental summaries for drug studies
- validation against hand-labeled sessions

### 6. Mature deployment and scaling

For broader use, the app would likely need:

- background jobs for long video analyses
- better session persistence
- more robust artifact indexing
- clearer deployment separation between frontend and backend

## Bottom line

This application is currently a usable rat-video screening tool with a strong architecture for future upgrades. The live, dependable path today is:

- upload video
- extract motion features
- optionally enrich with DeepLabCut pose
- classify behavior with heuristics unless trained checkpoints are present
- return annotated artifacts and review summaries

The codebase is already pointed toward a more serious research pipeline built around richer DLC pose estimation, trained temporal behavior models, SimBA-style analysis, and condition-aware experimental reporting. The main unfinished work is not application wiring; it is model readiness, labeling scale, and validation.
