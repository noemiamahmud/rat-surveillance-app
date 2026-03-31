from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from .pose_analysis import analyze_dlc_csv
from .config import settings


def _status_line(name: str, ok: bool, detail: str) -> str:
    state = "OK" if ok else "MISSING"
    return f"[{state}] {name}: {detail}"


def doctor() -> int:
    checks: list[tuple[str, bool, str]] = []

    dlc_installed = importlib.util.find_spec("deeplabcut") is not None
    torch_installed = importlib.util.find_spec("torch") is not None
    checks.append(("deeplabcut", dlc_installed, "Python package importable" if dlc_installed else "Install `deeplabcut` in the backend environment"))
    checks.append(("torch", torch_installed, "Python package importable" if torch_installed else "Install `torch` in the backend environment"))

    dlc_config = Path(settings.DLC_CONFIG_PATH)
    checks.append(("dlc_config", dlc_config.exists(), str(dlc_config)))

    default_snapshot = (
        settings.DLC_PROJECT_DIR
        / "RatTracking-Noemia-2025-11-17"
        / "dlc-models-pytorch"
        / "iteration-0"
        / "RatTrackingNov17-trainset95shuffle1"
        / "train"
        / "snapshot-best-170.pt"
    )
    checks.append(("dlc_snapshot", default_snapshot.exists(), str(default_snapshot)))

    labeled_dataset = (
        settings.DLC_PROJECT_DIR
        / "RatTracking-Noemia-2025-11-17"
        / "training-datasets"
        / "iteration-0"
        / "UnaugmentedDataSet_RatTrackingNov17"
        / "CollectedData_Noemia.csv"
    )
    checks.append(("dlc_labels", labeled_dataset.exists(), str(labeled_dataset)))

    classifier_model_path = Path(settings.CLASSIFIER_MODEL_PATH).expanduser() if settings.CLASSIFIER_MODEL_PATH else None
    checks.append((
        "temporal_classifier_checkpoint",
        classifier_model_path.exists() if classifier_model_path else False,
        str(classifier_model_path) if classifier_model_path else "No CLASSIFIER_MODEL_PATH configured",
    ))

    simba_model_path = Path(settings.SIMBA_MODEL_PATH).expanduser() if settings.SIMBA_MODEL_PATH else None
    checks.append((
        "simba_checkpoint",
        simba_model_path.exists() if simba_model_path else False,
        str(simba_model_path) if simba_model_path else "No SIMBA_MODEL_PATH configured",
    ))

    llama_model = settings.BASE_DIR.parent / "models" / "qwen2.5-7b" / "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    checks.append(("local_llm_model", llama_model.exists(), str(llama_model)))

    for name, ok, detail in checks:
        print(_status_line(name, ok, detail))

    missing = [name for name, ok, _ in checks if not ok]
    if missing:
        print("\nNext blockers:")
        for name in missing:
            print(f"- {name}")
        return 1

    print("\nAll required external assets for the configured pipeline are present.")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="CLI utilities for the rat surveillance backend."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    analyze_parser = subparsers.add_parser(
        "analyze-dlc",
        help="Analyze a DeepLabCut CSV export for locomotor and stereotypy candidate metrics.",
    )
    analyze_parser.add_argument("dlc_csv_path", help="Path to a DeepLabCut CSV export.")
    analyze_parser.add_argument("--fps", type=float, default=None, help="Video frames per second.")
    analyze_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON path for saving the analysis summary.",
    )

    subparsers.add_parser(
        "doctor",
        help="Check whether the local environment has the assets required for DLC and trained classifiers.",
    )
    args = parser.parse_args()

    if args.command == "doctor":
        raise SystemExit(doctor())

    summary = analyze_dlc_csv(args.dlc_csv_path, fps=args.fps)
    payload = json.dumps(summary, indent=2)

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)


if __name__ == "__main__":
    main()
