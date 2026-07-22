from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .client import DashScopeClient, ModelResponse
from .config import AgentConfig
from .intervals import Interval
from .prompts import (
    ANALYZER_SYSTEM,
    COMPARE_SYSTEM,
    analyzer_prompt,
    compare_prompt,
)
from .state import AgentState, AnomalyMark, TraceLogger
from .visualization import render_compare, render_interval


ALLOWED_ANOMALY_TYPES = {"point", "contextual", "frequency", "trend", "shapelet", "unknown"}


@dataclass
class ToolEnvironment:
    series: np.ndarray
    reference_series: np.ndarray
    reference_intervals: list[Interval]
    reference_source: str
    run_dir: Path
    client: DashScopeClient
    config: AgentConfig
    state: AgentState
    trace: TraceLogger

    def _normalize_confidence(self, value: Any) -> float:
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            confidence = 0.0
        if 1.0 < confidence <= 4.0:
            confidence /= 4.0
        return min(1.0, max(0.0, confidence))

    def _candidate_marks(
        self,
        analysis: dict[str, Any],
        allowed: Interval,
        source: str,
    ) -> list[AnomalyMark]:
        candidates: list[AnomalyMark] = []
        raw_items = analysis.get("anomalies", [])
        if not isinstance(raw_items, list):
            return candidates
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                start = int(round(float(item.get("start"))))
                end = int(round(float(item.get("end", start))))
            except (TypeError, ValueError):
                continue
            start = max(allowed.start, min(start, allowed.end))
            end = max(start, min(end, allowed.end))
            kind = str(item.get("anomaly_type", "unknown")).strip().lower()
            if kind not in ALLOWED_ANOMALY_TYPES:
                kind = "unknown"
            candidates.append(
                AnomalyMark(
                    interval=Interval(start, end),
                    confidence=self._normalize_confidence(item.get("confidence", 0.0)),
                    anomaly_type=kind,
                    evidence=str(item.get("evidence", ""))[:1200],
                    source=source,
                )
            )
        self.state.analyzer_candidates.extend(candidates)
        return candidates

    def _log_model(self, event: str, response: ModelResponse, extra: dict[str, Any]) -> None:
        self.trace.log(
            event,
            {
                **extra,
                "response": response.content,
                "usage": response.usage,
                "request_id": response.request_id,
                "model": response.model,
            },
        )

    def plot(self, start: int, end: int) -> dict[str, Any]:
        start_value, end_value = sorted((max(0, int(start)), max(0, int(end))))
        interval = Interval(start_value, end_value).clamp(len(self.series))
        if interval.length < self.config.min_plot_points:
            center = (interval.start + interval.end) // 2
            half = self.config.min_plot_points // 2
            interval = Interval(max(0, center - half), min(len(self.series) - 1, center + half))
        plot_index = len(self.state.plot_history)
        path = self.run_dir / "plots" / f"plot_{plot_index:03d}_{interval.start}_{interval.end}.png"
        render_interval(
            self.series,
            interval,
            path,
            f"{self.state.dataset}/{self.state.sample_id} - test interval",
            self.config.plot_max_points,
        )
        response = self.client.call_json(
            ANALYZER_SYSTEM,
            analyzer_prompt(
                self.state.dataset,
                self.state.sample_id,
                interval.start,
                interval.end,
                self.state.dataset_context,
            ),
            [path],
        )
        candidates = self._candidate_marks(response.content, interval, f"plot:{path.name}")
        record = {
            "tool": "Plot",
            "interval": interval.to_list(),
            "path": str(path),
            "description": str(response.content.get("description", ""))[:2000],
            "candidate_count": len(candidates),
        }
        self.state.add_observed(interval)
        self.state.plot_history.append(record)
        self._log_model("tool_plot_observation", response, {"plot": record})
        return {
            "status": "ok",
            "tool": "Plot",
            "interval": interval.to_list(),
            "description": record["description"],
            "anomalies": [candidate.to_dict() for candidate in candidates],
        }

    def compare(self, start: int, end: int, reference_index: int = 0) -> dict[str, Any]:
        start_value, end_value = sorted((max(0, int(start)), max(0, int(end))))
        test_interval = Interval(start_value, end_value).clamp(len(self.series))
        if not self.reference_intervals:
            return {"status": "error", "tool": "Compare", "message": "No reference candidates"}
        reference_index = max(0, min(int(reference_index), len(self.reference_intervals) - 1))
        reference = self.reference_intervals[reference_index]
        plot_index = len(self.state.plot_history)
        path = self.run_dir / "plots" / f"compare_{plot_index:03d}_{test_interval.start}_{test_interval.end}.png"
        render_compare(
            self.series,
            test_interval,
            self.reference_series,
            reference,
            path,
            "Panel 1 - test interval",
            f"Panel 2 - label-free reference candidate ({self.reference_source})",
            self.config.plot_max_points,
        )
        response = self.client.call_json(
            COMPARE_SYSTEM,
            compare_prompt(
                test_interval.start,
                test_interval.end,
                reference.start,
                reference.end,
                self.reference_source,
            ),
            [path],
        )
        candidates = self._candidate_marks(response.content, test_interval, f"compare:{path.name}")
        record = {
            "tool": "Compare",
            "interval": test_interval.to_list(),
            "reference_interval": reference.to_list(),
            "reference_source": self.reference_source,
            "path": str(path),
            "description": str(response.content.get("description", ""))[:2000],
            "candidate_count": len(candidates),
        }
        self.state.add_observed(test_interval)
        self.state.plot_history.append(record)
        self._log_model("tool_compare_observation", response, {"plot": record})
        return {
            "status": "ok",
            "tool": "Compare",
            "interval": test_interval.to_list(),
            "reference_interval": reference.to_list(),
            "description": record["description"],
            "anomalies": [candidate.to_dict() for candidate in candidates],
        }

    def mark(
        self,
        start: int,
        end: int,
        confidence: float,
        anomaly_type: str,
        evidence: str,
    ) -> dict[str, Any]:
        start_value, end_value = sorted((max(0, int(start)), max(0, int(end))))
        interval = Interval(start_value, end_value).clamp(len(self.series))
        if not any(interval.overlaps(observed) for observed in self.state.observed):
            return {
                "status": "error",
                "tool": "Mark",
                "message": "Mark rejected because the interval has not been observed",
            }
        normalized_confidence = self._normalize_confidence(confidence)
        if self.config.mode == "optimized":
            supported = any(interval.overlaps(candidate.interval) for candidate in self.state.analyzer_candidates)
            if not supported:
                return {
                    "status": "error",
                    "tool": "Mark",
                    "message": "Mark rejected by evidence gate: no analyzer candidate overlaps this interval",
                }
            if normalized_confidence < self.config.confidence_threshold:
                return {
                    "status": "error",
                    "tool": "Mark",
                    "message": "Mark rejected by confidence gate",
                }
        kind = str(anomaly_type).lower().strip()
        if kind not in ALLOWED_ANOMALY_TYPES:
            kind = "unknown"
        mark = AnomalyMark(
            interval=interval,
            confidence=normalized_confidence,
            anomaly_type=kind,
            evidence=str(evidence)[:1200],
            source="executor",
        )
        self.state.add_mark(mark, self.config.max_marks)
        payload = {"status": "ok", "tool": "Mark", "mark": mark.to_dict()}
        self.trace.log("tool_mark", payload)
        return payload

    def finish(self, reason: str) -> dict[str, Any]:
        payload = {"status": "ok", "tool": "Finish", "reason": str(reason)[:1200]}
        self.trace.log("tool_finish", payload)
        return payload

    def execute(self, tool: str, args: dict[str, Any]) -> dict[str, Any]:
        if tool == "Plot":
            return self.plot(args.get("start", 0), args.get("end", len(self.series) - 1))
        if tool == "Compare":
            return self.compare(
                args.get("start", 0),
                args.get("end", len(self.series) - 1),
                args.get("reference_index", 0),
            )
        if tool == "Mark":
            return self.mark(
                args.get("start", 0),
                args.get("end", args.get("start", 0)),
                args.get("confidence", 0.0),
                args.get("anomaly_type", "unknown"),
                args.get("evidence", ""),
            )
        if tool == "Finish":
            return self.finish(args.get("reason", "sub-task complete"))
        return {"status": "error", "message": f"Unknown tool: {tool}"}
