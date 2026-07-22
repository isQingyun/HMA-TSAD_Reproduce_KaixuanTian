from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from .intervals import Interval, coverage_fraction, merge_intervals


def interval_overlap_fraction(first: Interval, second: Interval) -> float:
    """Return overlap relative to the shorter interval.

    This treats slightly shifted or differently scaled views of the same event
    as one focus, while leaving disjoint parts of the series independent.
    """

    return first.intersection_length(second) / max(1, min(first.length, second.length))


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
    event_guard_points: int = 32
    analysis_windows: dict[str, int] = field(default_factory=dict)
    estimated_period: int | None = None
    reference_memory: dict[str, Any] = field(default_factory=dict)
    overview: list[Interval] = field(default_factory=list)
    observed: list[Interval] = field(default_factory=list)
    plot_history: list[dict[str, Any]] = field(default_factory=list)
    task_history: list[dict[str, Any]] = field(default_factory=list)
    marks: list[AnomalyMark] = field(default_factory=list)
    analyzer_candidates: list[AnomalyMark] = field(default_factory=list)
    dismissed_candidates: set[tuple[int, int, str, str]] = field(default_factory=set)
    active_strategy: str = ""
    novelty_redirects: int = 0
    blocked_repeated_visual_actions: int = 0
    exploration_targets: int = 0
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

    @staticmethod
    def candidate_signature(candidate: AnomalyMark) -> tuple[int, int, str, str]:
        return (
            candidate.interval.start,
            candidate.interval.end,
            candidate.anomaly_type,
            candidate.source,
        )

    @staticmethod
    def observation_key(source: str) -> str:
        """Normalize candidate/Mark sources to the visual model call that produced them."""

        key = str(source)
        while key.startswith("executor:"):
            key = key[len("executor:") :]
        return key.split(":origin=", 1)[0]

    def independent_detail_support(self, interval: Interval, overlap_threshold: float = 0.5) -> int:
        sources = {
            self.observation_key(candidate.source)
            for candidate in self.analyzer_candidates
            if candidate.source.startswith("detail:")
            and interval_overlap_fraction(interval, candidate.interval) >= overlap_threshold
        }
        return len(sources)

    def resolved_event_intervals(self) -> list[Interval]:
        """Return small event guards, never full plot windows.

        Plot reuse and event resolution are deliberately separate concerns.  A
        Mark closes only itself and a configurable nearby margin, so another
        anomaly already visible in the same chart remains actionable.
        """

        margin = max(0, int(self.event_guard_points))
        return merge_intervals(
            [
                Interval(
                    max(0, mark.interval.start - margin),
                    min(self.series_length - 1, mark.interval.end + margin),
                )
                for mark in self.marks
            ]
        )

    def is_resolved_event(
        self,
        interval: Interval,
        anomaly_type: str | None = None,
        overlap_threshold: float = 0.5,
    ) -> bool:
        """Whether a candidate is the same event as an accepted Mark.

        This method is for event candidates, never plot windows.  Contained or
        strongly overlapping candidates are therefore one event.  Disjoint
        candidates are treated as the same event only inside the small event
        guard, and only when their anomaly types agree.
        """

        margin = max(0, int(self.event_guard_points))
        for mark in self.marks:
            overlap = interval.intersection_length(mark.interval)
            if overlap / max(1, min(interval.length, mark.interval.length)) >= overlap_threshold:
                return True
            if overlap:
                continue
            gap = max(interval.start, mark.interval.start) - min(interval.end, mark.interval.end) - 1
            same_type = anomaly_type is None or anomaly_type == mark.anomaly_type
            if same_type and 0 <= gap <= margin:
                return True
        return False

    def candidate_is_handled(self, candidate: AnomalyMark, overlap_threshold: float = 0.5) -> bool:
        return (
            self.candidate_signature(candidate) in self.dismissed_candidates
            or self.is_resolved_event(candidate.interval, candidate.anomaly_type, overlap_threshold)
        )

    def dismiss_candidate(self, candidate: AnomalyMark) -> None:
        self.dismissed_candidates.add(self.candidate_signature(candidate))

    def next_unobserved_interval(
        self,
        desired: int | None = None,
        avoid: list[Interval] | None = None,
    ) -> Interval | None:
        """Choose a window from the largest remaining detail-coverage gap."""

        window = max(1, min(int(desired or self.analysis_window_points), self.series_length))
        gaps: list[Interval] = []
        cursor = 0
        for observed in merge_intervals([*self.observed, *(avoid or [])]):
            if cursor < observed.start:
                gaps.append(Interval(cursor, observed.start - 1))
            cursor = max(cursor, observed.end + 1)
        if cursor < self.series_length:
            gaps.append(Interval(cursor, self.series_length - 1))
        if not gaps:
            return None
        gap = max(gaps, key=lambda item: (item.length, -item.start))
        if gap.length <= window:
            return gap
        start = gap.start + (gap.length - window) // 2
        return Interval(start, start + window - 1)

    def attention_summary(self) -> dict[str, Any]:
        """Summarize whether visual effort was spread across distinct regions."""

        actions: list[tuple[str, Interval]] = []
        for record in self.plot_history:
            tool = str(record.get("tool", ""))
            if tool == "Plot" and record.get("resolution") != "detail":
                continue
            # A Compare may render a long period-aligned context around a tiny
            # target.  Attention identity follows the requested event focus,
            # not the automatically expanded 1P/3P/5P display interval.
            raw = (
                record.get("requested_interval", record.get("interval", []))
                if tool == "Compare"
                else record.get("interval", [])
            )
            if tool not in {"Plot", "Compare"} or not isinstance(raw, list) or len(raw) != 2:
                continue
            actions.append((tool, Interval(int(raw[0]), int(raw[1])).clamp(self.series_length)))

        clusters: list[list[Interval]] = []
        for _, interval in actions:
            matches = [
                index
                for index, cluster in enumerate(clusters)
                if any(interval_overlap_fraction(interval, member) >= 0.5 for member in cluster)
            ]
            if not matches:
                clusters.append([interval])
                continue
            first = matches[0]
            clusters[first].append(interval)
            for index in reversed(matches[1:]):
                clusters[first].extend(clusters.pop(index))

        signatures = [(tool, interval.start, interval.end) for tool, interval in actions]
        exact_repeats = len(signatures) - len(set(signatures))
        same_tool_near_repeats = sum(
            any(
                previous_tool == tool
                and interval_overlap_fraction(interval, previous_interval) >= 0.5
                for previous_tool, previous_interval in actions[:index]
            )
            for index, (tool, interval) in enumerate(actions)
        )
        return {
            "detail_visual_actions": len(actions),
            "unique_focus_regions": len(clusters),
            "actions_per_focus": round(len(actions) / max(1, len(clusters)), 6),
            "exact_repeat_actions": exact_repeats,
            "same_tool_near_repeat_actions": same_tool_near_repeats,
            "blocked_repeated_visual_actions": self.blocked_repeated_visual_actions,
            "novelty_redirects": self.novelty_redirects,
            "exploration_targets": self.exploration_targets,
            "resolved_event_intervals": [item.to_list() for item in self.resolved_event_intervals()],
            "dismissed_candidate_count": len(self.dismissed_candidates),
        }

    def planner_view(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "sample_id": self.sample_id,
            "series_length": self.series_length,
            "dataset_context": self.dataset_context,
            "analysis_window_points": self.analysis_window_points,
            "analysis_windows": self.analysis_windows,
            "estimated_period": self.estimated_period,
            "reference_memory": self.reference_memory,
            "detail_coverage": round(self.coverage, 6),
            "overview_coverage": round(self.overview_coverage, 6),
            "overview_intervals": [item.to_list() for item in self.overview],
            "observed_intervals": [item.to_list() for item in self.observed],
            "recent_plots": self.plot_history[-6:],
            "subtask_history": self.task_history[-6:],
            "current_marks": [item.to_dict() for item in self.marks],
            "resolved_event_intervals": [item.to_list() for item in self.resolved_event_intervals()],
            "attention_policy": (
                "An accepted Mark resolves only the same event and a small nearby guard. "
                "Rendered views are deduplicated separately; distinct candidates in one stored view remain actionable."
            ),
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
