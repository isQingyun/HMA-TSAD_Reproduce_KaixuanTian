from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .intervals import Interval, coverage_fraction, merge_intervals


@dataclass
class AnomalyMark:
    interval: Interval
    confidence: float
    anomaly_type: str
    evidence: str
    source: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": self.interval.start,
            "end": self.interval.end,
            "confidence": self.confidence,
            "anomaly_type": self.anomaly_type,
            "evidence": self.evidence,
            "source": self.source,
        }


@dataclass
class AgentState:
    dataset: str
    sample_id: str
    series_length: int
    dataset_context: dict[str, Any]
    observed: list[Interval] = field(default_factory=list)
    plot_history: list[dict[str, Any]] = field(default_factory=list)
    task_history: list[dict[str, Any]] = field(default_factory=list)
    marks: list[AnomalyMark] = field(default_factory=list)
    analyzer_candidates: list[AnomalyMark] = field(default_factory=list)
    finished: bool = False

    @property
    def coverage(self) -> float:
        return coverage_fraction(self.observed, self.series_length)

    def add_observed(self, interval: Interval) -> None:
        self.observed = merge_intervals([*self.observed, interval.clamp(self.series_length)])

    def add_mark(self, mark: AnomalyMark, max_marks: int) -> None:
        if len(self.marks) >= max_marks:
            return
        self.marks.append(mark)

    def planner_view(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "series_length": self.series_length,
            "dataset_context": self.dataset_context,
            "coverage": round(self.coverage, 6),
            "observed_intervals": [item.to_list() for item in self.observed],
            "recent_plots": self.plot_history[-6:],
            "subtask_history": self.task_history[-6:],
            "current_marks": [item.to_dict() for item in self.marks],
        }


class TraceLogger:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: str, payload: dict[str, Any]) -> None:
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event": event,
            "payload": payload,
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def save_json(path: str | Path, payload: dict[str, Any] | list[Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(output)

