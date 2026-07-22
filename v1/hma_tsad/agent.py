from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .client import DashScopeClient, ModelResponse
from .config import AgentConfig
from .data import TimeSeriesSample, robust_standardize, select_reference_intervals
from .intervals import Interval, merge_intervals
from .prompts import (
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    REFLECTION_SYSTEM,
    executor_prompt,
    planner_prompt,
    reflection_prompt,
)
from .state import AgentState, AnomalyMark, TraceLogger, save_json
from .summarizer import TaskSummarizer
from .tools import ALLOWED_ANOMALY_TYPES, ToolEnvironment
from .visualization import render_interval


def _model_record(response: ModelResponse) -> dict[str, Any]:
    return {
        "response": response.content,
        "usage": response.usage,
        "request_id": response.request_id,
        "model": response.model,
    }


def _target_from_goal(goal: dict[str, Any], length: int) -> Interval:
    targets = goal.get("target_intervals", [])
    if isinstance(targets, list) and targets and isinstance(targets[0], (list, tuple)) and len(targets[0]) >= 2:
        try:
            start = int(round(float(targets[0][0])))
            end = int(round(float(targets[0][1])))
            return Interval(max(0, start), max(0, end)).clamp(length)
        except (TypeError, ValueError):
            pass
    return Interval(0, length - 1)


class Planner:
    def __init__(
        self,
        client: DashScopeClient,
        config: AgentConfig,
        state: AgentState,
        trace: TraceLogger,
    ) -> None:
        self.client = client
        self.config = config
        self.state = state
        self.trace = trace
        self.goal_signatures: set[tuple[str, int, int]] = set()

    def _fallback(self, step: int) -> dict[str, Any]:
        if not self.state.observed:
            target = Interval(0, self.state.series_length - 1)
            strategy = "overview"
            goal_text = "Inspect a global overview and characterize the dominant pattern and deviations."
        elif self.state.analyzer_candidates:
            candidate = max(self.state.analyzer_candidates, key=lambda item: item.confidence)
            padding = max(64, candidate.interval.length)
            target = Interval(
                max(0, candidate.interval.start - padding),
                min(self.state.series_length - 1, candidate.interval.end + padding),
            )
            strategy = "compare" if step > 0 else "zoom"
            goal_text = "Zoom into the strongest candidate and compare it with a label-free reference."
        else:
            chunk = max(64, self.state.series_length // max(1, self.config.max_planner_steps))
            start = min(self.state.series_length - 1, step * chunk)
            target = Interval(start, min(self.state.series_length - 1, start + chunk - 1))
            strategy = "coverage"
            goal_text = "Inspect an unverified region to avoid missing anomalies."
        return {
            "thought": "Deterministic safe fallback plan.",
            "goal": goal_text,
            "strategy": strategy,
            "target_intervals": [target.to_list()],
            "reference_index": 0,
            "done": False,
        }

    def next_goal(self, step: int) -> dict[str, Any]:
        try:
            response = self.client.call_json(
                PLANNER_SYSTEM,
                planner_prompt(
                    self.state.planner_view(),
                    step,
                    self.config.max_planner_steps,
                    self.config.mode == "optimized",
                ),
                max_tokens=700,
            )
            goal = dict(response.content)
            self.trace.log("planner_response", {"step": step, **_model_record(response)})
        except Exception as exc:  # keep the experiment resumable on transient model failures
            self.trace.log("planner_error", {"step": step, "error": str(exc)})
            goal = self._fallback(step)

        if not self.state.observed:
            goal.update(
                {
                    "goal": "Inspect the global overview, establish the dominant pattern, and identify candidate deviations.",
                    "strategy": "overview",
                    "target_intervals": [[0, self.state.series_length - 1]],
                    "done": False,
                }
            )
        elif self.config.mode == "optimized" and self.state.analyzer_candidates:
            strongest = max(self.state.analyzer_candidates, key=lambda item: item.confidence)
            padding = max(256, strongest.interval.length * 8)
            zoom = Interval(
                max(0, strongest.interval.start - padding),
                min(self.state.series_length - 1, strongest.interval.end + padding),
            )
            goal.update(
                {
                    "thought": "Coordinate-refinement guard: zoom around the strongest overview candidate before accepting it.",
                    "goal": "Compare a wide zoom around the strongest global candidate with a label-free normal reference; refine coordinates and reject normal-cycle morphology.",
                    "strategy": "compare",
                    "target_intervals": [zoom.to_list()],
                    "reference_index": step % 2,
                    "done": False,
                }
            )
            self.trace.log("planner_coordinate_refinement_guard", {"replacement": goal})
        target = _target_from_goal(goal, self.state.series_length)
        strategy = str(goal.get("strategy", "overview")).lower()
        signature = (strategy, target.start, target.end)
        if self.config.mode == "optimized" and signature in self.goal_signatures:
            goal = self._fallback(step)
            target = _target_from_goal(goal, self.state.series_length)
            signature = (str(goal.get("strategy")), target.start, target.end)
            self.trace.log("planner_repetition_guard", {"replacement": goal})
        self.goal_signatures.add(signature)
        goal["target_intervals"] = [target.to_list()]
        goal["reference_index"] = max(0, int(goal.get("reference_index", 0) or 0))
        goal["done"] = bool(goal.get("done", False)) and bool(self.state.observed)
        return goal


class ReActExecutor:
    def __init__(
        self,
        client: DashScopeClient,
        config: AgentConfig,
        state: AgentState,
        environment: ToolEnvironment,
        trace: TraceLogger,
    ) -> None:
        self.client = client
        self.config = config
        self.state = state
        self.environment = environment
        self.trace = trace
        self.summarizer = TaskSummarizer(client, trace)

    def _fallback_action(
        self,
        goal: dict[str, Any],
        observations: list[dict[str, Any]],
    ) -> dict[str, Any]:
        target = _target_from_goal(goal, self.state.series_length)
        if not observations:
            tool = "Compare" if str(goal.get("strategy", "")).lower() == "compare" else "Plot"
            args: dict[str, Any] = {"start": target.start, "end": target.end}
            if tool == "Compare":
                args["reference_index"] = int(goal.get("reference_index", 0))
            return {"thought": "Acquire visual evidence for the target interval.", "tool": tool, "args": args}
        for observation in reversed(observations):
            anomalies = observation.get("anomalies", []) if isinstance(observation, dict) else []
            if anomalies:
                item = max(anomalies, key=lambda value: float(value.get("confidence", 0.0)))
                already_marked = any(
                    mark.interval.overlaps(Interval(int(item["start"]), int(item["end"])))
                    for mark in self.state.marks
                )
                if not already_marked:
                    return {
                        "thought": "Register the strongest visually supported candidate.",
                        "tool": "Mark",
                        "args": item,
                    }
        return {"thought": "No further evidence-based action is needed.", "tool": "Finish", "args": {"reason": "goal complete"}}

    def run(self, goal: dict[str, Any]) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        for tool_step in range(self.config.max_executor_steps):
            try:
                response = self.client.call_json(
                    EXECUTOR_SYSTEM,
                    executor_prompt(
                        goal,
                        self.state.planner_view(),
                        observations,
                        tool_step,
                        self.config.max_executor_steps,
                        len(self.environment.reference_intervals),
                    ),
                    max_tokens=650,
                )
                action = dict(response.content)
                self.trace.log(
                    "executor_thought_action",
                    {"tool_step": tool_step, "goal": goal, **_model_record(response)},
                )
            except Exception as exc:
                self.trace.log("executor_error", {"tool_step": tool_step, "error": str(exc)})
                action = self._fallback_action(goal, observations)

            tool = str(action.get("tool", "")).strip().title()
            if tool not in {"Plot", "Compare", "Mark", "Finish"}:
                action = self._fallback_action(goal, observations)
                tool = action["tool"]
            args = action.get("args", {})
            if not isinstance(args, dict):
                args = {}
            if (
                self.config.mode == "optimized"
                and not observations
                and str(goal.get("strategy", "")).lower() == "compare"
            ):
                target = _target_from_goal(goal, self.state.series_length)
                action = {
                    "thought": "Optimized executor enforces the planned reference comparison before marking.",
                    "tool": "Compare",
                    "args": {
                        "start": target.start,
                        "end": target.end,
                        "reference_index": int(goal.get("reference_index", 0)),
                    },
                }
                tool = "Compare"
                args = action["args"]
            signature = json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False)
            seen[signature] = seen.get(signature, 0) + 1
            if seen[signature] > 1:
                self.trace.log("executor_repetition", {"tool_step": tool_step, "action": action})
                action = self._fallback_action(goal, observations)
                tool = action["tool"]
                args = action["args"]

            observation = self.environment.execute(tool, args)
            actions.append({"thought": str(action.get("thought", "")), "tool": tool, "args": args})
            observations.append(observation)
            self.trace.log(
                "executor_observation",
                {"tool_step": tool_step, "tool": tool, "observation": observation},
            )
            if tool == "Finish":
                break

        task_summary = self.summarizer.summarize(goal, actions, observations)
        summary = {
            "goal": str(goal.get("goal", "")),
            "strategy": str(goal.get("strategy", "")),
            "summary": task_summary,
            "actions": [item["tool"] for item in actions],
            "new_observations": len(observations),
            "coverage_after": round(self.state.coverage, 6),
            "mark_count_after": len(self.state.marks),
            "last_observation": observations[-1] if observations else {},
        }
        self.trace.log("subtask_summary", summary)
        return summary


class HMATSADAgent:
    def __init__(
        self,
        client: DashScopeClient,
        config: AgentConfig,
        run_dir: str | Path,
    ) -> None:
        self.client = client
        self.config = config
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.trace = TraceLogger(self.run_dir / "agent_trace.jsonl")

    def _promote_candidates(self, state: AgentState) -> None:
        if state.marks or self.config.mode != "optimized":
            return
        candidates = sorted(state.analyzer_candidates, key=lambda item: item.confidence, reverse=True)
        for candidate in candidates:
            if candidate.confidence < self.config.confidence_threshold:
                continue
            state.add_mark(
                AnomalyMark(
                    interval=candidate.interval,
                    confidence=candidate.confidence,
                    anomaly_type=candidate.anomaly_type,
                    evidence=candidate.evidence,
                    source="optimized-evidence-promotion",
                ),
                self.config.max_marks,
            )
        self.trace.log("optimized_candidate_promotion", {"mark_count": len(state.marks)})

    def _reflect(self, series: np.ndarray, state: AgentState) -> None:
        if not self.config.enable_reflection:
            return
        self._promote_candidates(state)
        if not state.marks:
            self.trace.log("reflection_skipped", {"reason": "no candidates"})
            return
        candidates = sorted(state.marks, key=lambda item: item.confidence, reverse=True)[:6]
        images: list[Path] = []
        metadata: list[dict[str, Any]] = []
        for index, candidate in enumerate(candidates):
            padding = max(128, candidate.interval.length * 4)
            view = Interval(
                max(0, candidate.interval.start - padding),
                min(len(series) - 1, candidate.interval.end + padding),
            )
            path = self.run_dir / "plots" / f"reflection_{index:02d}_{view.start}_{view.end}.png"
            render_interval(series, view, path, f"Reflection candidate {index + 1}", self.config.plot_max_points)
            images.append(path)
            item = candidate.to_dict()
            item["visible_range"] = view.to_list()
            metadata.append(item)
        try:
            response = self.client.call_json(
                REFLECTION_SYSTEM,
                reflection_prompt(metadata),
                images,
                max_tokens=1200,
            )
            self.trace.log("reflection_response", {"candidates": metadata, **_model_record(response)})
            verified_raw = response.content.get("verified_anomalies", [])
            verified: list[AnomalyMark] = []
            visible = [Interval(*item["visible_range"]) for item in metadata]
            for item in verified_raw if isinstance(verified_raw, list) else []:
                if not isinstance(item, dict):
                    continue
                try:
                    raw_start = max(0, int(round(float(item.get("start")))))
                    raw_end = max(0, int(round(float(item.get("end", item.get("start"))))))
                    start, end = sorted((raw_start, raw_end))
                    interval = Interval(start, end).clamp(len(series))
                    confidence = float(item.get("confidence", 0.0))
                    if 1.0 < confidence <= 4.0:
                        confidence /= 4.0
                    elif confidence > 4.0:
                        confidence /= 100.0
                    confidence = min(1.0, max(0.0, confidence))
                except (TypeError, ValueError):
                    continue
                if confidence < self.config.confidence_threshold:
                    continue
                if not any(interval.overlaps(view) for view in visible):
                    continue
                kind = str(item.get("anomaly_type", "unknown")).lower()
                if kind not in ALLOWED_ANOMALY_TYPES:
                    kind = "unknown"
                verified.append(
                    AnomalyMark(
                        interval=interval,
                        confidence=confidence,
                        anomaly_type=kind,
                        evidence=str(item.get("evidence", ""))[:1200],
                        source="multi-scale-reflection",
                    )
                )
            state.marks = verified[: self.config.max_marks]
        except Exception as exc:
            self.trace.log("reflection_error", {"error": str(exc), "kept_original_marks": True})

    def _final_predictions(self, state: AgentState) -> list[dict[str, Any]]:
        groups = merge_intervals([mark.interval for mark in state.marks], gap=self.config.merge_gap)
        predictions: list[dict[str, Any]] = []
        for group in groups:
            members = [mark for mark in state.marks if mark.interval.overlaps(group)]
            strongest = max(members, key=lambda item: item.confidence)
            predictions.append(
                {
                    "start": group.start,
                    "end": group.end,
                    "confidence": strongest.confidence,
                    "anomaly_type": strongest.anomaly_type,
                    "evidence": strongest.evidence,
                    "source": strongest.source,
                }
            )
        return predictions

    def run(self, sample: TimeSeriesSample) -> dict[str, Any]:
        series, normalization = robust_standardize(sample.values)
        reference_series, reference_intervals, reference_source = select_reference_intervals(sample)
        dataset_context = {
            "metadata": sample.metadata,
            "reference_source": reference_source,
            "normalization": "median/MAD robust standardization",
            "labels_available_to_agent": False,
        }
        state = AgentState(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            series_length=sample.length,
            dataset_context=dataset_context,
        )
        environment = ToolEnvironment(
            series=series,
            reference_series=reference_series,
            reference_intervals=reference_intervals,
            reference_source=reference_source,
            run_dir=self.run_dir,
            client=self.client,
            config=self.config,
            state=state,
            trace=self.trace,
        )
        planner = Planner(self.client, self.config, state, self.trace)
        executor = ReActExecutor(self.client, self.config, state, environment, self.trace)
        self.trace.log(
            "run_start",
            {
                "dataset": sample.dataset,
                "sample_id": sample.sample_id,
                "series_length": sample.length,
                "mode": self.config.mode,
                "agent_config": asdict(self.config),
                "reference_source": reference_source,
                "reference_intervals": [item.to_list() for item in reference_intervals],
            },
        )

        for planner_step in range(self.config.max_planner_steps):
            goal = planner.next_goal(planner_step)
            if goal.get("done"):
                self.trace.log("planner_finish", {"step": planner_step, "goal": goal})
                break
            summary = executor.run(goal)
            state.task_history.append(summary)

        self._reflect(series, state)
        predictions = self._final_predictions(state)
        state.finished = True
        result = {
            "schema_version": 1,
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "mode": self.config.mode,
            "model": self.client.config.model,
            "series_length": sample.length,
            "normalization": normalization,
            "reference_source": reference_source,
            "reference_intervals": [item.to_list() for item in reference_intervals],
            "coverage": state.coverage,
            "predictions": predictions,
            "observed_intervals": [item.to_list() for item in state.observed],
            "subtask_history": state.task_history,
            "label_policy": "ground-truth labels were not loaded during inference",
        }
        save_json(self.run_dir / "predictions.json", result)
        self.trace.log("run_complete", {"prediction_count": len(predictions), "coverage": state.coverage})
        return result
