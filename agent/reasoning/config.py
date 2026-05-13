from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PACKAGED_DP_SR_JAR_PATH = (
    Path(__file__).resolve().parent / "resources" / "dp-sr" / "DP-sr.jar"
)

DP_SR_FLAGS: tuple[str, ...] = (
    "--print-non-empty",
    "--t-format=sec",
    "--t-unit=sec",
    "--windows-unit=sec",
    "--now-format=sec",
    "--json-output",
)
DP_SR_DEBUG_FLAGS: tuple[str, ...] = (
    "--print-extended-log",
    "--print-reasoning-info",
    "--print-operators-info",
)


@dataclass(frozen=True)
class DpSrConfig:
    rules_path: Path | str
    read_timeout_s: float
    startup_timeout_s: float
    dpsr_debug: bool
    jar_path: Path = PACKAGED_DP_SR_JAR_PATH

    def __post_init__(self) -> None:
        object.__setattr__(self, "rules_path", Path(self.rules_path).resolve())

    def build_command(self, *, dp_src_endpoint: tuple[str, int]) -> list[str]:
        command = [
            "java",
            "-jar",
            str(self.jar_path),
            f"--program={self.rules_path}",
            f"--src-hostname={dp_src_endpoint[0]}",
            f"--src-port={dp_src_endpoint[1]}",
            *DP_SR_FLAGS,
        ]
        if self.dpsr_debug:
            command.extend(DP_SR_DEBUG_FLAGS)
        return command

    def check_runtime_requirements(self) -> None:
        if not self.jar_path.is_file():
            raise FileNotFoundError(f"DP-SR jar not found: {self.jar_path}")
        if not self.rules_path.is_file():
            raise FileNotFoundError(f"DP-SR rules file not found: {self.rules_path}")
