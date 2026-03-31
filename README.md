# Rat Behavior Intelligence

Rat Behavior Intelligence is a browser-based rat pose and behavior analysis app. A user uploads a rat video or records a short clip from the frontend, the backend runs a staged rodent behavior pipeline, and the app returns an annotated playback, timestamped behavior events, an AI observer chat feed, and a short run summary.

## What the program does

The current pipeline is designed to be practical and interpretable:

- accepts uploaded rat videos from the frontend
- supports browser webcam capture for quick testing
- extracts temporal movement, turning, confinement, and motion features frame by frame
- classifies rat-centric states such as `exploration`, `locomotor_burst`, `hesitation`, `freezing_candidate`, `grooming_candidate`, `turning_pattern`, `resting`, and `stereotypy_candidate`
- uses DeepLabCut pose enrichment when the local DLC environment is available
- uses richer rat-pose geometry when extra keypoints are present, including ears, spine points, paws, and tail segments
- overlays predicted behavior labels directly onto the output video
- generates timestamped behavior events and an AI observer chat feed
- returns a short run summary plus review-priority guidance

This is framed as an AI behavior classifier, but the current upload workflow is still a heuristic-first system rather than a fully trained cross-species model. It is best used as a screening and review tool.

## Current architecture

- [backend/app/main.py](/Users/noemiamahmud/rat-surveillance-app/backend/app/main.py): FastAPI app and static serving
- [backend/app/routes.py](/Users/noemiamahmud/rat-surveillance-app/backend/app/routes.py): upload and analysis endpoints
- [backend/app/video_annotation.py](/Users/noemiamahmud/rat-surveillance-app/backend/app/video_annotation.py): rat behavior classifier, event generation, and annotated video output
- [backend/app/pose_analysis.py](/Users/noemiamahmud/rat-surveillance-app/backend/app/pose_analysis.py): pose-trace analysis utilities for DeepLabCut CSVs
- [frontend/index.html](/Users/noemiamahmud/rat-surveillance-app/frontend/index.html): upload UI
- [frontend/app.js](/Users/noemiamahmud/rat-surveillance-app/frontend/app.js): frontend analysis workflow

## Run locally

```bash
cd /Users/noemiamahmud/rat-surveillance-app/backend
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

Open:
[http://127.0.0.1:8000](http://127.0.0.1:8000)

Copy the backend environment template before running a non-heuristic pipeline:

```bash
cd /Users/noemiamahmud/rat-surveillance-app/backend
cp .env.example .env
python -m app.cli doctor
```

`python -m app.cli doctor` checks whether the local environment can actually use:

- the DeepLabCut Python package
- the repo's DLC config and trained snapshot
- an optional temporal-classifier checkpoint
- an optional SimBA checkpoint
- the bundled local LLM artifact

If `doctor` reports `deeplabcut` missing, the upload pipeline will run in motion-fallback mode even though the repo already contains a DLC project and trained pose snapshots.

## Manual DLC labeling

The DeepLabCut labeling environment is set up in the local conda env at:
- `/Users/noemiamahmud/miniconda3/envs/rat-surveillance`

Use the repo helper instead of calling DLC by hand:

```bash
cd /Users/noemiamahmud/rat-surveillance-app/backend
bash run_dlc_labeling.sh --help
```

For a new rat video you want to label:

```bash
cd /Users/noemiamahmud/rat-surveillance-app/backend
bash run_dlc_labeling.sh prepare-video /absolute/path/to/your/rat_video.mp4
```

That command does two things:
- adds the video to the existing DLC project
- extracts labeling frames into `dlc_project/RatTracking-Noemia-2025-11-17/labeled-data/<video_stem>/`

Then open the labeling GUI for that extracted frame folder:

```bash
cd /Users/noemiamahmud/rat-surveillance-app/backend
bash run_dlc_labeling.sh label --image-folder rat_video
```

Replace `rat_video` with the actual MP4 filename stem. For example, `rat_video_3.mp4` uses:

```bash
bash run_dlc_labeling.sh label --image-folder rat_video_3
```

After you finish labeling and save your points, generate DLC's label-check images:

```bash
cd /Users/noemiamahmud/rat-surveillance-app/backend
bash run_dlc_labeling.sh check-labels
```

### Your part

The remaining manual work is the actual annotation pass in the GUI:
- run `prepare-video` for each new MP4
- run `label --image-folder <video_stem>`
- click all 16 body parts on each extracted frame
- save the labels in the napari window before closing it
- run `check-labels`
- inspect the generated labeled previews under `dlc_project/.../labeled-data/*_labeled/`

The body parts in this project are:
- `nose`
- `left_ear`
- `right_ear`
- `neck`
- `shoulder`
- `upper_back`
- `spine_mid`
- `lower_back`
- `hip`
- `tail_base`
- `tail_mid`
- `tail_tip`
- `left_forepaw`
- `right_forepaw`
- `left_hindpaw`
- `right_hindpaw`

### Notes

- Run the labeling command from your normal desktop Terminal session, not a headless shell, because napari needs a GUI session.
- If you ever move or recreate the conda env, point the helper at a different Python with `DLC_PYTHON=/path/to/python bash run_dlc_labeling.sh ...`
- If you want to skip the separate extract step and immediately open labeling after extraction, use:

```bash
cd /Users/noemiamahmud/rat-surveillance-app/backend
bash run_dlc_labeling.sh prepare-video /absolute/path/to/your/rat_video.mp4 --label-now
```

## Outputs

Each analysis produces:

- an annotated browser-playable MP4
- the original uploaded video
- a `timeline.json` file with frame-level labels and bouts
- a short natural-language summary
- timestamped behavior events
- an AI observer chat transcript
- a low/medium/high review priority flag

## Models Used Right Now

The app does not currently use an LLM or embedding model for reasoning. The live AI observer feed is generated by a rule-based event interpreter over the behavior timeline.

If `LLM_PROVIDER=openai` and `OPENAI_API_KEY` is set in the backend environment, the final observer summary upgrades to a real OpenAI model through the Responses API using `OPENAI_MODEL` (default: `gpt-5.2-chat-latest`).

If `LLM_PROVIDER=llama_cpp`, the app can also use a **local `llama.cpp` OpenAI-compatible server** instead of OpenAI. In that case set:

```bash
export LLM_PROVIDER=llama_cpp
export LLM_BASE_URL=http://127.0.0.1:8001
export LLAMA_CPP_MODEL=your-loaded-model-name
```

The timestamped live event feed is still generated directly from the analysis pipeline so it can stream during processing; the LLM is used for the final observer summary layer.

When pose enrichment is available, the real neural network used for pose prediction is the local DeepLabCut model in this repo:

- DeepLabCut
- engine: PyTorch
- backbone: ResNet-50 GN
- head: heatmap-based keypoint prediction
- body parts: `nose`, `upper_back`, `lower_back`, `tail_base`, `tail_tip`

The pose-analysis layer is now prepared for a stronger retrained rat model that tracks:

- `nose`, `left_ear`, `right_ear`
- `neck`, `shoulder`, `upper_back`, `spine_mid`, `lower_back`, `hip`
- `tail_base`, `tail_mid`, `tail_tip`
- `left_forepaw`, `right_forepaw`, `left_hindpaw`, `right_hindpaw`

The retraining template for that richer rat skeleton lives at:
- [dlc_project/rat_pose_expanded_config_template.yaml](/Users/noemiamahmud/rat-surveillance-app/dlc_project/rat_pose_expanded_config_template.yaml)

This configuration comes from:
- [dlc_project/RatTracking-Noemia-2025-11-17/config.yaml](/Users/noemiamahmud/rat-surveillance-app/dlc_project/RatTracking-Noemia-2025-11-17/config.yaml)
- [dlc_project/RatTracking-Noemia-2025-11-17/dlc-models-pytorch/iteration-0/RatTrackingNov17-trainset95shuffle1/train/pytorch_config.yaml](/Users/noemiamahmud/rat-surveillance-app/dlc_project/RatTracking-Noemia-2025-11-17/dlc-models-pytorch/iteration-0/RatTrackingNov17-trainset95shuffle1/train/pytorch_config.yaml)

When DeepLabCut is not available for a run, the app falls back to a motion-based temporal rat behavior classifier and reports that choice in the UI.

## Running With llama.cpp Locally

One realistic local setup is:

1. Start a llama.cpp-compatible server that exposes an OpenAI-style API.
2. Point this app at that server with `LLM_PROVIDER=llama_cpp`.
3. Restart the FastAPI backend.

This project does not download or launch the model for you automatically; it expects your local llama.cpp server to already be running.

Artifacts are stored under:
- [backend/data/uploads](/Users/noemiamahmud/rat-surveillance-app/backend/data/uploads)
- [backend/data/processed/annotated_sessions](/Users/noemiamahmud/rat-surveillance-app/backend/data/processed/annotated_sessions)

## Accuracy and future strengthening

The strongest next upgrade is to move more of the classification onto pose-derived temporal features:

- run DeepLabCut consistently on uploaded rat videos
- retrain DeepLabCut with the expanded rat body schema so grooming, hesitation, turning, and posture changes are driven by more than a 5-point spine/tail chain
- train a temporal classifier on labeled rat behavior sequences
- improve the grooming, freezing, and stereotypy boundaries with human-reviewed clips
- add live event streaming for truly real-time analysis
- move longer analyses into background jobs for deployment

## Deployment note

This cannot run as GitHub Pages alone because video upload and annotation require a backend. The frontend can be deployed statically, but the analysis API needs a server or worker runtime.
