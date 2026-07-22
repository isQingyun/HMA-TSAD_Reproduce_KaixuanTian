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
    analysis_window_points: int = 256
    estimated_period: int | None = None
    reference_memory: dict[str, Any] = field(default_factory=dict)
    overview: list[Interval] = field(default_factory=list)
    observed: list[Interval] = field(default_factory=list)
    plot_history: list[dict[str, Any]] = field(default_factory=list)
    task_history: list[dict[str, Any]] = field(default_factory=list)
    marks: list[AnomalyMark] = field(default_factory=list)
    analyzer_candidates: list[AnomalyMark] = field(default_factory=list)
    finished: bool = False

    @property
    def coverage(self) -> float:
        return coverage_fraction(self.observed, self.series_length)

    @property
    def overview_coverage(self) -> float:
        return coverage_fraction(self.overview, self.series_length)

    def add_overview(self, interval: Interval) -> None:
        self.overview = merge_intervals([*self.overview, interval.clamp(self.series_length)])

    def add_observed(self, interval: Interval) -> None:
        self.observed = merge_intervals([*self.observed, interval.clamp(self.series_length)])

    def add_mark(self, mark: AnomalyMark, max_marks: int) -> str:
        overlapping = [index for index, existing in enumerate(self.marks) if existing.interval.overlaps(mark.interval)]
        if overlapping:
            candidates = [(self.marks[index], index) for index in overlapping] + [(mark, -1)]

            def rank(item: tuple[AnomalyMark, int]) -> tuple[int, float, int]:
                candidate, _ = item
                source_priority = 2 if "compare" in candidate.source else 1 if "plot" in candidate.source else 0
                return source_priority, candidate.confidence, -candidate.interval.length

            winner, winner_index = max(candidates, key=rank)
            first = overlapping[0]
            self.marks[first] = winner
            for index in reversed(overlapping[1:]):
                del self.marks[index]
            return "updated" if winner_index == -1 else "unchanged"
        if len(self.marks) >= max(0, max_marks):
            return "limit"
        self.marks.append(mark)
        return "added"

    def planner_view(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "series_length": self.series_length,
            "dataset_context": self.dataset_context,
            "analysis_window_points": self.analysis_window_points,
            "estimated_period": self.estimated_period,
            "reference_memory": self.reference_memory,
            "detail_coverage": round(self.coverage, 6),
            "overview_coverage": round(self.overview_coverage, 6),
            "overview_intervals": [item.to_list() for item in self.overview],
            "observed_intervals": [item.to_list() for item in self.observed],
            "recent_plots": self.plot_history[-6:],
            "subtask_history": self.task_history[-6:],
            "current_marks": [item.to_dict() for item in self.marks],
            "analyzer_candidates": [item.to_dict() for item in self.analyzer_candidates[-12:]],
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
