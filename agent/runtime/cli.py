from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from ..model.types import Area


DEFAULT_GEMINI_MODEL = "gemini-2.5-flash"
DEFAULT_YOLO_MODEL = "./models/yolo26x.pt"
DEFAULT_VLM_MAX_FRAMES = 32


@dataclass(frozen=True)
class RuntimePaths:
    video_path: Path
    areas_path: Path | None
    rules_path: Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stream reasoning agent for object-based anomaly detection."
    )
    parser.add_argument(
        "--scenario",
        help=(
            "Scenario directory containing one input video file, 'rules.lp', "
            "and optionally 'areas.json'."
        ),
    )
    parser.add_argument(
        "--video-path",
        help="Path to the input video file. Overrides scenario video discovery.",
    )
    parser.add_argument(
        "--areas",
        help="Optional path to the areas JSON file. Overrides scenario areas.json.",
    )
    parser.add_argument(
        "--rules",
        help="Path to the rules LP file. Overrides scenario rules.lp.",
    )
    parser.add_argument(
        "--yolo-model",
        default=DEFAULT_YOLO_MODEL,
        help="Path to the YOLO weights file.",
    )
    parser.add_argument(
        "--yolo-device",
        default=_default_yolo_device(),
        help="YOLO inference device. Defaults to 'cuda' when available, otherwise 'cpu'.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory in which runtime logs and generated artifacts are stored.",
    )
    parser.add_argument(
        "--vlm-api-key",
        default=None,
        help="Google AI Studio Gemini API key. Falls back to GEMINI_API_KEY or GOOGLE_API_KEY.",
    )
    parser.add_argument(
        "--vlm-model",
        default=DEFAULT_GEMINI_MODEL,
        help="Google AI Studio Gemini model for VLM validation.",
    )
    parser.add_argument(
        "--vlm-input-mode",
        choices=("video", "frames"),
        default="video",
        help="Visual evidence format sent to the VLM backend.",
    )
    parser.add_argument(
        "--save-vlm-input",
        action="store_true",
        help="Save sampled frames or video clips sent to the VLM under <output-dir>/vlm_input/.",
    )
    parser.add_argument(
        "--save-vlm-debug-clips",
        action="store_true",
        dest="save_vlm_input",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--vlm-workers",
        type=_positive_int,
        default=1,
        help="Number of background VLM workers.",
    )
    parser.add_argument(
        "--vlm-max-frames",
        type=_positive_int,
        default=DEFAULT_VLM_MAX_FRAMES,
        help="Maximum sampled frames per candidate when input mode chosen is frames.",
    )
    parser.add_argument(
        "--dpsr-debug",
        "--dp-sr-debug",
        action="store_true",
        dest="dpsr_debug",
        help="Enable verbose DP-SR process output in agent.log.",
    )
    return parser


def resolve_runtime_paths(args: argparse.Namespace) -> RuntimePaths:
    scenario_dir = _resolve_scenario_dir(args.scenario)
    scenario_video = None
    if scenario_dir is not None and not args.video_path:
        scenario_video = _discover_scenario_video(scenario_dir)
    scenario_areas = _scenario_file(scenario_dir, "areas.json")
    scenario_rules = _scenario_file(scenario_dir, "rules.lp")

    video_path = _pick_required_path(
        cli_value=args.video_path,
        discovered_value=scenario_video,
        label="video file",
        scenario_dir=scenario_dir,
        expected_name=None,
    )
    areas_path = _pick_optional_path(
        cli_value=args.areas,
        discovered_value=scenario_areas,
        label="Areas file",
    )
    rules_path = _pick_required_path(
        cli_value=args.rules,
        discovered_value=scenario_rules,
        label="rules file",
        scenario_dir=scenario_dir,
        expected_name="rules.lp",
    )
    return RuntimePaths(
        video_path=_require_file(video_path, label="Video file"),
        areas_path=areas_path,
        rules_path=_require_file(rules_path, label="Rules file"),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from ..logging.events import EventLogger
    from ..reasoning.config import DpSrConfig
    from ..reasoning.reasoner import DpSrReasoner
    from ..vision.perception import Perception
    from ..vlm import (
        VLMConfig,
        VisualReasoner,
        build_visual_language_classifier,
    )
    from .application import Agent

    event_logger = None
    reasoner = None
    perception = None
    visual_language_model = None
    try:
        runtime_paths = resolve_runtime_paths(args)
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        event_logger = EventLogger(log_path=str(output_dir / "agent.log"))
        areas_text = (
            "<none>" if runtime_paths.areas_path is None else str(runtime_paths.areas_path)
        )
        event_logger.agent(
            (
                "startup "
                f"video={runtime_paths.video_path} "
                f"areas={areas_text} "
                f"rules={runtime_paths.rules_path} "
                f"output_dir={output_dir} "
                f"vlm_input_mode={args.vlm_input_mode} "
                f"vlm_max_frames={args.vlm_max_frames} "
                f"vlm_workers={args.vlm_workers} "
                f"save_vlm_input={args.save_vlm_input} "
                f"dpsr_debug={args.dpsr_debug}"
            ),
        )

        if runtime_paths.areas_path is None:
            areas = []
            event_logger.info("Loaded 0 areas", console=True)
        else:
            areas = Perception.load_areas_from_json(str(runtime_paths.areas_path))
            event_logger.info(_format_loaded_areas(areas), console=True)

        perception = Perception(
            video_path=str(runtime_paths.video_path),
            model_path=args.yolo_model,
            yolo_device=args.yolo_device,
            areas=areas,
            logger=event_logger,
        )
        reasoner = DpSrReasoner(
            config=DpSrConfig(
                rules_path=runtime_paths.rules_path,
                read_timeout_s=0.02,
                startup_timeout_s=10.0,
                dpsr_debug=args.dpsr_debug,
            ),
            logger=event_logger,
        )
        reasoner.connect()
        vlm_classifier = build_visual_language_classifier(
            gemini_api_key=_resolve_vlm_api_key(args.vlm_api_key),
            gemini_model=args.vlm_model,
            vlm_input_mode=args.vlm_input_mode,
        )
        visual_language_model = VisualReasoner(
            config=VLMConfig(
                video_path=runtime_paths.video_path,
                vlm_input_mode=args.vlm_input_mode,
                save_vlm_input=args.save_vlm_input,
                output_dir=output_dir,
                max_workers=args.vlm_workers,
                max_frames_per_clip=args.vlm_max_frames,
            ),
            classifier=vlm_classifier,
            logger=event_logger,
        )
        agent = Agent(
            perception=perception,
            reasoner=reasoner,
            visual_language_model=visual_language_model,
            logger=event_logger,
            log_facts=False,
        )
        agent.run()
        return 0
    except Exception as exc:
        if event_logger is not None:
            try:
                event_logger.error(str(exc), console=True)
            except RuntimeError:
                print(f"[error] {exc}", file=sys.stderr)
        else:
            print(f"[error] {exc}", file=sys.stderr)
        return 1
    finally:
        if visual_language_model is not None:
            visual_language_model.close()
        if perception is not None:
            perception.close()
        if reasoner is not None:
            reasoner.close()
        if event_logger is not None:
            event_logger.close()


def _resolve_scenario_dir(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None

    scenario_dir = Path(raw_path)
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Scenario directory not found: {scenario_dir}")
    if not scenario_dir.is_dir():
        raise ValueError(f"Scenario path is not a directory: {scenario_dir}")
    return scenario_dir


def _scenario_file(scenario_dir: Path | None, filename: str) -> Path | None:
    if scenario_dir is None:
        return None
    path = scenario_dir / filename
    return path if path.is_file() else None


def _pick_required_path(
    *,
    cli_value: str | None,
    discovered_value: Path | None,
    label: str,
    scenario_dir: Path | None,
    expected_name: str | None,
) -> Path:
    if cli_value:
        return Path(cli_value)
    if discovered_value is not None:
        return discovered_value

    if scenario_dir is not None and expected_name is not None:
        raise FileNotFoundError(
            f"Scenario '{scenario_dir}' is missing required file '{expected_name}'."
        )
    raise ValueError(
        f"Missing required {label}. Provide it explicitly or use --scenario."
    )


def _pick_optional_path(
    *,
    cli_value: str | None,
    discovered_value: Path | None,
    label: str,
) -> Path | None:
    if cli_value:
        return _require_file(Path(cli_value), label=label)
    return discovered_value


def _require_file(path: Path, *, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    return path


def _discover_scenario_video(scenario_dir: Path) -> Path:
    scenario_metadata = {"areas.json", "rules.lp"}
    candidates = sorted(
        path
        for path in scenario_dir.iterdir()
        if path.is_file() and path.name not in scenario_metadata
    )
    if not candidates:
        raise FileNotFoundError(
            f"Scenario '{scenario_dir}' must contain exactly one input video file; "
            "found none."
        )
    if len(candidates) > 1:
        candidate_names = ", ".join(path.name for path in candidates)
        raise ValueError(
            f"Scenario '{scenario_dir}' must contain exactly one input video file; "
            f"found multiple candidates: {candidate_names}."
        )
    return candidates[0]


def _resolve_vlm_api_key(cli_value: str | None) -> str | None:
    return cli_value or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")


def _format_loaded_areas(areas: Sequence[Area]) -> str:
    if not areas:
        return "Loaded 0 areas"

    area_text = ", ".join(
        f"{area.area_id} ({area.area_type})"
        for area in areas
    )
    return f"Loaded {len(areas)} areas: {area_text}"


def _default_yolo_device() -> str:
    try:
        import torch
    except Exception:
        return "cpu"
    try:
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


def _positive_int(raw_value: str) -> int:
    value = int(raw_value)
    if value < 1:
        raise argparse.ArgumentTypeError("value must be at least 1")
    return value
