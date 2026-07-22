from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    model: str = "qwen3-vl-plus"
    api_base: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    api_key_file: str = ".secrets/dashscope_api_key"
    timeout_seconds: int = 120
    max_retries: int = 3
    temperature: float = 0.1
    top_p: float = 0.3
    max_tokens: int = 1400


@dataclass(frozen=True)
class AgentConfig:
    mode: str = "baseline"
    max_planner_steps: int = 6
    max_executor_steps: int = 6
    max_marks: int = 12
    min_plot_points: int = 64
    plot_max_points: int = 9000
    enable_window_slicing: bool = True
    slice_overlap_ratio: float = 0.5
    slice_vote_threshold: float = 0.5
    slice_vote_aggregation: str = "sum"
    target_coverage: float = 0.95
    enable_reflection: bool = False
    enable_reference_learning: bool = True
    reference_count: int = 4
    reference_window_cycles: float = 3.0
    reference_cycle_plan: tuple[float, ...] = (1.0, 3.0, 3.0, 5.0)
    adaptive_compare_cycles: tuple[float, ...] = (1.0, 3.0, 5.0)
    reference_min_points: int = 256
    reference_max_points: int = 2048
    confidence_threshold: float = 0.55
    merge_gap: int = 2
    focus_overlap_threshold: float = 0.5
    event_guard_points: int = 32
    coverage_confidence_threshold: float = 0.9

    def __post_init__(self) -> None:
        if self.plot_max_points <= 0:
            raise ValueError("plot_max_points must be positive")
        if not 0.0 <= self.slice_overlap_ratio < 1.0:
            raise ValueError("slice_overlap_ratio must be in [0, 1)")
        if not 0.0 < self.slice_vote_threshold <= 1.0:
            raise ValueError("slice_vote_threshold must be in (0, 1]")
        if self.slice_vote_aggregation not in {"mean", "sum"}:
            raise ValueError("slice_vote_aggregation must be 'mean' or 'sum'")

    @classmethod
    def for_mode(cls, mode: str, **overrides: Any) -> "AgentConfig":
        if mode not in {"baseline", "optimized"}:
            raise ValueError(f"Unsupported mode: {mode}")
        for key in ("reference_cycle_plan", "adaptive_compare_cycles"):
            if key in overrides:
                overrides[key] = tuple(float(value) for value in overrides[key])
        defaults: dict[str, Any] = {
            "mode": mode,
            "enable_reflection": mode == "optimized",
        }
        defaults.update(overrides)
        return cls(**defaults)


@dataclass(frozen=True)
class ExperimentConfig:
    data_root: str
    output_root: str
    seed: int
    samples: dict[str, list[str]]
    model: ModelConfig
    agent: AgentConfig

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_experiment_config(path: str | Path, mode: str | None = None) -> ExperimentConfig:
    config_path = Path(path)
    raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    model = ModelConfig(**raw.get("model", {}))
    agent_raw = dict(raw.get("agent", {}))
    yaml_mode = agent_raw.pop("mode", "baseline")
    selected_mode = mode or yaml_mode
    agent = AgentConfig.for_mode(selected_mode, **agent_raw)
    return ExperimentConfig(
        data_root=str(raw.get("data_root", "tsad_datasets")),
        output_root=str(raw.get("output_root", "artifacts")),
        seed=int(raw.get("seed", 20260719)),
        samples={str(k): [str(v) for v in values] for k, values in raw["samples"].items()},
        model=model,
        agent=agent,
    )
