from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .client import DashScopeClient, ModelResponse
from .config import AgentConfig
from .data import shape_distance
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
    reference_cycles: list[float]
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
        if not np.isfinite(confidence):
            confidence = 0.0
        if 1.0 < confidence <= 4.0:
            confidence /= 4.0
        elif confidence > 4.0:
            confidence /= 100.0
        return min(1.0, max(0.0, confidence))

    @staticmethod
    def _fit_interval(interval: Interval, desired: int, series_length: int) -> Interval:
        desired = max(1, min(int(desired), series_length))
        center = (interval.start + interval.end) // 2
        start = max(0, center - desired // 2)
        end = start + desired - 1
        if end >= series_length:
            end = series_length - 1
            start = max(0, end - desired + 1)
        return Interval(start, end)

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
            start, end = sorted((start, end))
            if end < allowed.start or start > allowed.end:
                # Clamping a completely out-of-view hallucination to the nearest
                # chart boundary manufactures a false anomaly at that boundary.
                continue
            start = max(allowed.start, min(start, allowed.end))
            end = max(start, min(end, allowed.end))
            kind = str(item.get("anomaly_type", "unknown")).strip().lower()
            if kind not in ALLOWED_ANOMALY_TYPES:
                kind = "unknown"
            evidence_parts = [str(item.get("evidence", "")).strip()]
            if item.get("counterevidence"):
                evidence_parts.append(f"Counterevidence checked: {str(item['counterevidence']).strip()}")
            if item.get("boundary_rationale"):
                evidence_parts.append(f"Boundary rationale: {str(item['boundary_rationale']).strip()}")
            candidates.append(
                AnomalyMark(
                    interval=Interval(start, end),
                    confidence=self._normalize_confidence(item.get("confidence", 0.0)),
                    anomaly_type=kind,
                    evidence=" | ".join(part for part in evidence_parts if part)[:1200],
                    source=source,
                )
            )
        self.state.analyzer_candidates.extend(candidates)
        return candidates

    def _reference_profiles(self) -> dict[int, dict[str, Any]]:
        profiles = self.state.reference_memory.get("reference_profiles", [])
        return {
            int(item.get("reference_index")): item
            for item in profiles
            if isinstance(item, dict) and str(item.get("reference_index", "")).lstrip("-").isdigit()
        }

    def _reference_cycles_at(self, index: int, profile: dict[str, Any]) -> float:
        if 0 <= index < len(self.reference_cycles):
            try:
                cycles = float(self.reference_cycles[index])
                if np.isfinite(cycles) and cycles > 0.0:
                    return cycles
            except (TypeError, ValueError):
                pass
        try:
            cycles = float(profile.get("window_cycles", self.config.reference_window_cycles))
        except (TypeError, ValueError):
            cycles = float(self.config.reference_window_cycles)
        return cycles if np.isfinite(cycles) and cycles > 0.0 else 3.0

    def _score_reference(
        self,
        test_interval: Interval,
        reference_index: int,
        profile: dict[str, Any],
        cycles: float,
    ) -> dict[str, Any]:
        test_segment = self.series[test_interval.start : test_interval.end + 1]
        reference = self.reference_intervals[reference_index]
        reference_segment = self.reference_series[reference.start : reference.end + 1]
        distance = shape_distance(test_segment, reference_segment)
        reliability = self._normalize_confidence(profile.get("reliability", 0.5))
        match_score = distance + 0.25 * (1.0 - reliability)
        return {
            "reference_index": reference_index,
            "window_cycles": float(cycles),
            "shape_distance": round(distance, 6),
            "reliability": round(reliability, 6),
            "match_score": round(match_score, 6),
            # High only when a reliable same-scale normal reference differs.
            "scale_deviation_score": round(distance * reliability, 6),
        }

    @staticmethod
    def _coerce_cycle_choice(value: Any, available: list[float]) -> float | None:
        text = str(value).strip().lower().replace("cycles", "").replace("cycle", "").replace("p", "")
        try:
            parsed = float(text)
        except (TypeError, ValueError):
            return None
        return next((item for item in available if abs(item - parsed) < 1e-6), None)

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
            interval = self._fit_interval(interval, self.config.min_plot_points, len(self.series))
        resolution = (
            "overview"
            if interval.length > self.state.analysis_window_points * 2
            else "detail"
        )
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
                {
                    **self.state.dataset_context,
                    "resolution": resolution,
                    "analysis_window_points": self.state.analysis_window_points,
                    "reference_memory": self.state.reference_memory,
                },
            ),
            [path],
        )
        candidates = self._candidate_marks(response.content, interval, f"{resolution}:plot:{path.name}")
        record = {
            "tool": "Plot",
            "interval": interval.to_list(),
            "resolution": resolution,
            "path": str(path),
            "description": str(response.content.get("description", ""))[:2000],
            "candidate_count": len(candidates),
        }
        if resolution == "overview":
            self.state.add_overview(interval)
        else:
            self.state.add_observed(interval)
        self.state.plot_history.append(record)
        self._log_model("tool_plot_observation", response, {"plot": record})
        return {
            "status": "ok",
            "tool": "Plot",
            "interval": interval.to_list(),
            "resolution": resolution,
            "description": record["description"],
            "anomalies": [candidate.to_dict() for candidate in candidates],
        }

    def compare(self, start: int, end: int, reference_index: Any = -1) -> dict[str, Any]:
        start_value, end_value = sorted((max(0, int(start)), max(0, int(end))))
        requested_interval = Interval(start_value, end_value).clamp(len(self.series))
        if not self.reference_intervals:
            return {"status": "error", "tool": "Compare", "message": "No reference candidates"}
        try:
            requested_index = int(reference_index)
        except (TypeError, ValueError):
            requested_index = -1
        explicit_reference = 0 <= requested_index < len(self.reference_intervals)
        candidate_indices = [requested_index] if explicit_reference else list(range(len(self.reference_intervals)))
        profiles = self._reference_profiles()
        plot_index = len(self.state.plot_history)
        comparisons: list[dict[str, Any]] = []
        public_comparisons: list[dict[str, Any]] = []
        for image_position, index in enumerate(candidate_indices, start=1):
            reference = self.reference_intervals[index]
            profile = profiles.get(index, {})
            cycles = self._reference_cycles_at(index, profile)
            scale_key = f"{cycles:g}"
            desired = int(self.state.analysis_windows.get(scale_key, reference.length))
            test_interval = self._fit_interval(requested_interval, desired, len(self.series))
            score = self._score_reference(test_interval, index, profile, cycles)
            path = self.run_dir / "plots" / (
                f"compare_{plot_index:03d}_{cycles:g}P_ref{index}_"
                f"{test_interval.start}_{test_interval.end}.png"
            )
            render_compare(
                self.series,
                test_interval,
                self.reference_series,
                reference,
                path,
                f"Panel 1 - test query ({cycles:g}P)",
                f"Panel 2 - reference {index} ({cycles:g}P, {self.reference_source})",
                self.config.plot_max_points,
            )
            comparison = {
                "reference_index": index,
                "cycles": cycles,
                "test_interval": test_interval,
                "reference_interval": reference,
                "profile": profile,
                "score": score,
                "path": path,
            }
            comparisons.append(comparison)
            public_comparisons.append(
                {
                    "image_position": image_position,
                    "window_cycles": cycles,
                    "test_range": test_interval.to_list(),
                    "reference_index": index,
                    "reference_range": reference.to_list(),
                    "reference_profile": profile,
                    "numeric_evidence": score,
                }
            )

        available_scales = list(dict.fromkeys(float(item["cycles"]) for item in comparisons))
        scale_scores: list[dict[str, Any]] = []
        for cycles in available_scales:
            same_scale = [item for item in comparisons if abs(float(item["cycles"]) - cycles) < 1e-6]
            best_match = min(same_scale, key=lambda item: float(item["score"]["match_score"]))
            scale_scores.append(
                {
                    "window_cycles": cycles,
                    "best_reference_index": int(best_match["reference_index"]),
                    "nearest_normal_distance": float(best_match["score"]["shape_distance"]),
                    "reference_reliability": float(best_match["score"]["reliability"]),
                    "scale_deviation_score": float(best_match["score"]["scale_deviation_score"]),
                }
            )
        fallback_scale = (
            float(comparisons[0]["cycles"])
            if explicit_reference
            else float(max(scale_scores, key=lambda item: item["scale_deviation_score"])["window_cycles"])
        )
        preselection = {
            "mode": "catalog-requested" if explicit_reference else "automatic-multiscale-probe",
            "requested_index": requested_index,
            "fallback_scale_cycles": fallback_scale,
            "scale_scores": scale_scores,
            "reference_scores": [item["score"] for item in comparisons],
        }
        response = self.client.call_json(
            COMPARE_SYSTEM,
            compare_prompt(
                public_comparisons,
                self.reference_source,
                str(self.state.reference_memory.get("normal_pattern", "")),
                preselection,
            ),
            [item["path"] for item in comparisons],
        )

        model_scale = self._coerce_cycle_choice(response.content.get("selected_scale_cycles"), available_scales)
        if explicit_reference:
            selected_scale = float(comparisons[0]["cycles"])
            selection_mode = "catalog-requested"
        elif model_scale is not None:
            selected_scale = model_scale
            selection_mode = "model-multiscale"
        else:
            selected_scale = fallback_scale
            selection_mode = "deterministic-multiscale-fallback"
        eligible = [item for item in comparisons if abs(float(item["cycles"]) - selected_scale) < 1e-6]
        try:
            model_reference_index = int(response.content.get("selected_reference_index", -1))
        except (TypeError, ValueError):
            model_reference_index = -1
        selected_comparison = next(
            (item for item in eligible if int(item["reference_index"]) == model_reference_index),
            min(eligible, key=lambda item: float(item["score"]["match_score"])),
        )
        selected_reference_index = int(selected_comparison["reference_index"])
        test_interval = selected_comparison["test_interval"]
        reference = selected_comparison["reference_interval"]
        path = selected_comparison["path"]
        selection = {
            **preselection,
            "mode": selection_mode,
            "model_selected_scale_cycles": model_scale,
            "selected_scale_cycles": selected_scale,
            "model_selected_reference_index": model_reference_index,
            "selected_index": selected_reference_index,
        }
        candidates = self._candidate_marks(
            response.content,
            test_interval,
            f"detail:compare:{selected_scale:g}P:ref{selected_reference_index}:{path.name}",
        )
        record = {
            "tool": "Compare",
            "requested_interval": requested_interval.to_list(),
            "interval": test_interval.to_list(),
            "query_intervals": [item["test_range"] for item in public_comparisons],
            "selected_scale_cycles": selected_scale,
            "reference_interval": reference.to_list(),
            "reference_index": selected_reference_index,
            "reference_source": self.reference_source,
            "reference_selection": selection,
            "path": str(path),
            "paths": [str(item["path"]) for item in comparisons],
            "description": str(response.content.get("description", ""))[:2000],
            "candidate_count": len(candidates),
        }
        self.state.add_observed(test_interval)
        self.state.plot_history.append(record)
        self._log_model("tool_compare_observation", response, {"plot": record})
        return {
            "status": "ok",
            "tool": "Compare",
            "requested_interval": requested_interval.to_list(),
            "interval": test_interval.to_list(),
            "query_intervals": [item["test_range"] for item in public_comparisons],
            "selected_scale_cycles": selected_scale,
            "reference_interval": reference.to_list(),
            "reference_index": selected_reference_index,
            "reference_selection": selection,
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
        source: str = "model",
    ) -> dict[str, Any]:
        start_value, end_value = sorted((max(0, int(start)), max(0, int(end))))
        interval = Interval(start_value, end_value).clamp(len(self.series))
        if not any(observed.start <= interval.start and interval.end <= observed.end for observed in self.state.observed):
            return {
                "status": "error",
                "tool": "Mark",
                "message": "Mark rejected because the interval has not been observed",
            }
        normalized_confidence = self._normalize_confidence(confidence)
        if self.config.mode == "optimized":
            supported = any(
                interval.overlaps(candidate.interval) and candidate.source.startswith("detail:")
                for candidate in self.state.analyzer_candidates
            )
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
        supporting_candidates = [
            candidate
            for candidate in self.state.analyzer_candidates
            if candidate.source.startswith("detail:") and candidate.interval.overlaps(interval)
        ]
        if supporting_candidates:
            supporting_candidates.sort(
                key=lambda candidate: (
                    1 if "compare" in candidate.source else 0,
                    candidate.confidence,
                ),
                reverse=True,
            )
            evidence_source = supporting_candidates[0].source
        else:
            evidence_source = str(source)[:240]
        mark = AnomalyMark(
            interval=interval,
            confidence=normalized_confidence,
            anomaly_type=kind,
            evidence=str(evidence)[:1200],
            source=f"executor:{evidence_source}",
        )
        state_action = self.state.add_mark(mark, self.config.max_marks)
        payload = {
            "status": "ok" if state_action != "limit" else "error",
            "tool": "Mark",
            "state_action": state_action,
            "mark": mark.to_dict(),
        }
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
                args.get("reference_index", -1),
            )
        if tool == "Mark":
            return self.mark(
                args.get("start", 0),
                args.get("end", args.get("start", 0)),
                args.get("confidence", 0.0),
                args.get("anomaly_type", "unknown"),
                args.get("evidence", ""),
                args.get("source", "model"),
            )
        if tool == "Finish":
            return self.finish(args.get("reason", "sub-task complete"))
        return {"status": "error", "message": f"Unknown tool: {tool}"}
