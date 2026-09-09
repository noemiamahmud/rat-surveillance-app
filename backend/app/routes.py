# backend/app/routes.py
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter, HTTPException, File, UploadFile, Form
from fastapi.responses import StreamingResponse

from .schemas import (
    CameraConnectRequest,
    CameraStatus,
    SessionStartRequest,
    SessionInfo,
    BehaviorTimeline,
    OfflineAnalysisRequest,
    PoseAnalysisResponse,
    SimBAAnalysisRequest,
    SimBAAnalysisResponse,
    VideoAnalysisRequest,
    UploadedAnalysisResponse,
)
from .video_stream import stream_manager
from .pose_analysis import analyze_dlc_csv
from .video_annotation import annotate_behavior_video, stream_behavior_analysis
from .simba_classifier import get_simba_classifier
from .condition_profiles import generate_condition_report
from .config import settings

router = APIRouter(prefix="/api")


# -------- Camera control --------

@router.post("/camera/connect", response_model=CameraStatus)
def connect_camera(payload: CameraConnectRequest):
    try:
        stream_manager.connect(rtsp_url=payload.rtsp_url)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))

    return CameraStatus(
        connected=True, rtsp_url=stream_manager.rtsp_url, running=stream_manager.running
    )


@router.post("/camera/disconnect", response_model=CameraStatus)
def disconnect_camera():
    stream_manager.disconnect()
    return CameraStatus(connected=False, rtsp_url=None, running=False)


@router.get("/camera/status", response_model=CameraStatus)
def camera_status():
    return CameraStatus(
        connected=stream_manager.capture is not None,
        rtsp_url=stream_manager.rtsp_url,
        running=stream_manager.running,
    )


# -------- Live video stream (MJPEG) --------

def mjpeg_generator():
    while True:
        frame = stream_manager.get_jpeg_frame()
        if frame is None:
            # No frame yet; yield a tiny pause
            import time as _time
            _time.sleep(0.1)
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        )


@router.get("/stream/live")
def live_stream():
    if not stream_manager.running:
        raise HTTPException(status_code=400, detail="Camera not connected.")
    return StreamingResponse(
        mjpeg_generator(), media_type="multipart/x-mixed-replace; boundary=frame"
    )


# -------- Sessions --------

@router.post("/sessions/start", response_model=SessionInfo)
def start_session(payload: SessionStartRequest):
    try:
        info = stream_manager.start_session(name=payload.name)
        return info
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/sessions/stop", response_model=SessionInfo)
def stop_session():
    info = stream_manager.stop_session()
    if info is None:
        raise HTTPException(status_code=400, detail="No active session.")
    return info


@router.get("/sessions", response_model=list[SessionInfo])
def list_sessions():
    return list(stream_manager.list_sessions().values())


# -------- Behavior --------

@router.get("/behavior/{session_id}", response_model=BehaviorTimeline)
def get_behavior(session_id: str):
    try:
        timeline = stream_manager.get_behavior_timeline(session_id)
        return timeline
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# -------- Offline pose analysis --------

@router.post("/analysis/from-dlc", response_model=PoseAnalysisResponse)
def analyze_from_dlc(payload: OfflineAnalysisRequest):
    try:
        analysis = analyze_dlc_csv(payload.dlc_csv_path, fps=payload.fps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PoseAnalysisResponse(
        subject_id=payload.subject_id,
        condition=payload.condition,
        source=payload.dlc_csv_path,
        bodyparts_used=analysis["bodyparts_used"],
        metrics=analysis["metrics"],
        bouts=analysis["bouts"],
        notes=analysis["notes"],
    )


@router.post("/analysis/upload-video", response_model=UploadedAnalysisResponse)
async def upload_video_for_analysis(
    file: UploadFile = File(...),
    condition: str | None = Form(default=None),
    classifier_type: str | None = Form(default=None),
):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    upload_id = uuid4().hex[:10]
    upload_path = settings.UPLOADS_DIR / f"{upload_id}{suffix}"

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    upload_path.write_bytes(contents)

    try:
        result = annotate_behavior_video(
            str(upload_path),
            str(settings.ANNOTATED_OUTPUTS_DIR),
            condition=condition,
            classifier_type=classifier_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    annotated_relative = Path(result["annotated_video_path"]).relative_to(settings.PROCESSED_DIR)
    timeline_relative = Path(result["timeline_path"]).relative_to(settings.PROCESSED_DIR)
    source_relative = Path(result["source_video_path"]).relative_to(settings.UPLOADS_DIR)

    return UploadedAnalysisResponse(
        analysis_id=result["analysis_id"],
        pipeline_mode=result["pipeline_mode"],
        model_stack=result["model_stack"],
        selected_models=result["selected_models"],
        source_video_url=f"/uploads/{source_relative.as_posix()}",
        annotated_video_url=f"/artifacts/{annotated_relative.as_posix()}",
        timeline_url=f"/artifacts/{timeline_relative.as_posix()}",
        fps=result["fps"],
        frame_count=result["frame_count"],
        duration_s=result["duration_s"],
        dominant_behavior=result["dominant_behavior"],
        behavior_share=result["behavior_share"],
        bouts=result["bouts"],
        behavior_events=result["behavior_events"],
        analyst_messages=result["analyst_messages"],
        analysis_summary=result["analysis_summary"],
        review_priority=result["review_priority"],
        classifier_mode=result["classifier_mode"],
        condition=result["condition"],
        condition_report=result["condition_report"],
        condition_anomalies=result["condition_anomalies"],
        notes=result["notes"],
    )


@router.post("/analysis/upload-video-stream")
async def upload_video_for_analysis_stream(
    file: UploadFile = File(...),
    condition: str | None = Form(default=None),
    classifier_type: str | None = Form(default=None),
):
    suffix = Path(file.filename or "upload.mp4").suffix or ".mp4"
    upload_id = uuid4().hex[:10]
    upload_path = settings.UPLOADS_DIR / f"{upload_id}{suffix}"

    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file was empty.")

    upload_path.write_bytes(contents)
    return StreamingResponse(
        stream_behavior_analysis(
            str(upload_path),
            str(settings.ANNOTATED_OUTPUTS_DIR),
            condition=condition,
            classifier_type=classifier_type,
        ),
        media_type="application/x-ndjson",
    )


@router.post("/analysis/simba-classify", response_model=SimBAAnalysisResponse)
def simba_classify(payload: SimBAAnalysisRequest):
    classifier = get_simba_classifier(classifier_type="simba")
    if classifier is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "SimBA classifier is not available. Configure CLASSIFIER_TYPE=simba "
                "and provide a trained SIMBA_MODEL_PATH checkpoint."
            ),
        )

    try:
        from .pose_analysis import load_dlc_csv

        df = load_dlc_csv(payload.dlc_csv_path)
        result = classifier.predict_with_quadrant_analysis(
            df,
            fps=payload.fps,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    dominant_behavior = max(result["behavior_share"], key=result["behavior_share"].get, default="monitoring")
    condition_report = generate_condition_report(
        result["behavior_share"],
        payload.condition,
        dominant_behavior,
    )
    notes = [
        "SimBA supervised classifier predictions were generated from DeepLabCut pose features.",
        "A trained SimBA checkpoint is required; when unavailable the main upload route falls back to heuristics.",
    ]

    return SimBAAnalysisResponse(
        subject_id=payload.subject_id,
        condition=payload.condition,
        source=payload.dlc_csv_path,
        classifier_mode="simba",
        predictions=result["predictions"],
        behavior_share=result["behavior_share"],
        quadrant_time=result["quadrant_time"],
        total_frames=result["total_frames"],
        duration_s=result["duration_s"],
        condition_report=condition_report,
        notes=notes,
    )


@router.post("/analysis/from-video", response_model=PoseAnalysisResponse)
def analyze_from_video(payload: VideoAnalysisRequest):
    try:
        from .dlc_runner import run_dlc_on_video

        pose_df = run_dlc_on_video(
            video_path=payload.video_path,
            config_path=payload.config_path,
            save_as_csv=True,
        )
        from .pose_analysis import summarize_pose_analysis

        analysis = summarize_pose_analysis(pose_df, fps=payload.fps)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ModuleNotFoundError as e:
        raise HTTPException(
            status_code=500,
            detail=(
                "DeepLabCut is not installed in the active environment. "
                f"Original error: {e}"
            ),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return PoseAnalysisResponse(
        subject_id=payload.subject_id,
        condition=payload.condition,
        source=payload.video_path,
        bodyparts_used=analysis["bodyparts_used"],
        metrics=analysis["metrics"],
        bouts=analysis["bouts"],
        notes=analysis["notes"],
    )
