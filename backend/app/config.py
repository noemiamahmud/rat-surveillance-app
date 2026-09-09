# backend/app/config.py
from pathlib import Path
from pydantic import BaseSettings


def _resolve_default_dlc_config(base_dir: Path) -> Path:
    dlc_project_dir = base_dir / "dlc_project"
    direct_config = dlc_project_dir / "config.yaml"
    if direct_config.exists():
        return direct_config

    nested_configs = sorted(dlc_project_dir.glob("*/config.yaml"))
    if nested_configs:
        return nested_configs[0]

    return direct_config


class Settings(BaseSettings):
    # Default RTSP URL – override via env var if you want
    WYZE_RTSP_URL: str = (
        "rtsp://username:password@CAMERA_IP/live"  # placeholder
    )

    # Data dirs
    BASE_DIR: Path = Path(__file__).resolve().parent.parent
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_VIDEOS_DIR: Path = DATA_DIR / "raw_videos"
    PROCESSED_DIR: Path = DATA_DIR / "processed"
    EXAMPLE_OUTPUTS_DIR: Path = DATA_DIR / "example_outputs"
    UPLOADS_DIR: Path = DATA_DIR / "uploads"
    ANNOTATED_OUTPUTS_DIR: Path = PROCESSED_DIR / "annotated_sessions"
    DLC_PROJECT_DIR: Path = Path(__file__).resolve().parent.parent.parent / "dlc_project"
    DLC_OUTPUT_ROOT: Path = PROCESSED_DIR / "dlc_outputs"
    DLC_CONFIG_PATH: Path = _resolve_default_dlc_config(Path(__file__).resolve().parent.parent.parent)
    LLM_PROVIDER: str = "none"
    LLM_BASE_URL: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_MODEL: str = "gpt-5.2-chat-latest"
    LLAMA_CPP_MODEL: str = "local-llama.cpp"
    CLASSIFIER_TYPE: str = "adaptive_hmm"
    CLASSIFIER_MODEL_PATH: str = ""
    CLASSIFIER_MODEL_TYPE: str = "lstm"
    SIMBA_MODEL_PATH: str = ""

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()

# Make sure dirs exist at import time
settings.RAW_VIDEOS_DIR.mkdir(parents=True, exist_ok=True)
settings.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
settings.EXAMPLE_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
settings.UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
settings.ANNOTATED_OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
settings.DLC_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)


# create later: .env file in backend/ with: WYZE_RTSP_URL=rtsp://youruser:yourpass@yourcamera/live
