from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Any

import numpy as np

from .client import DashScopeClient, ModelResponse
from .config import AgentConfig
from .data import (
    TimeSeriesSample,
    multiscale_window_points,
    reference_catalog as build_reference_catalog,
    robust_standardize,
    select_reference_intervals,
)
from .intervals import Interval, merge_intervals
from .prompts import (
    EXECUTOR_SYSTEM,
    PLANNER_SYSTEM,
    REFERENCE_SYSTEM,
    REFLECTION_SYSTEM,
    executor_prompt,
    planner_prompt,
    reference_prompt,
    reflection_prompt,
)
from .state import AgentState, AnomalyMark, TraceLogger, interval_overlap_fraction, save_json
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


def _window_around(interval: Interval, desired: int, length: int) -> Interval:
    desired = max(1, min(int(desired), length))
    center = (interval.start + interval.end) // 2
    start = max(0, center - desired // 2)
    end = start + desired - 1
    if end >= length:
        end = length - 1
        start = max(0, end - desired + 1)
    return Interval(start, end)


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
        self.goal_targets: list[tuple[str, Interval]] = []

    def _candidate_is_resolved(self, candidate: AnomalyMark) -> bool:
        return self.state.candidate_is_handled(
            candidate,
            self.config.focus_overlap_threshold,
        )

    def _unresolved_overview_candidate(self) -> AnomalyMark | None:
        candidates = [
            item
            for item in self.state.analyzer_candidates
            if item.source.startswith("overview:") and not self._candidate_is_resolved(item)
        ]
        unresolved = [
            item
            for item in candidates
            if not any(
                observed.start <= item.interval.start and item.interval.end <= observed.end
                for observed in self.state.observed
            )
        ]
        return max(unresolved, key=lambda item: item.confidence) if unresolved else None

    def _strongest_detail_candidate(self) -> AnomalyMark | None:
        compared = [
            Interval(*record["interval"])
            for record in self.state.plot_history
            if record.get("tool") == "Compare"
            and isinstance(record.get("interval"), list)
            and len(record["interval"]) == 2
        ]
        candidates = [
            item
            for item in self.state.analyzer_candidates
            if item.source.startswith("detail:plot:")
            and not self._candidate_is_resolved(item)
            and not any(view.start <= item.interval.start and item.interval.end <= view.end for view in compared)
        ]
        return max(candidates, key=lambda item: item.confidence) if candidates else None

    def _strongest_compare_candidate(self) -> AnomalyMark | None:
        """Return an unhandled event already supported by a stored Compare.

        This queue is what lets one Compare image yield several independent
        Marks without rendering or analyzing the same window again.
        """

        candidates = [
            item
            for item in self.state.analyzer_candidates
            if item.source.startswith("detail:compare:")
            and item.confidence >= self.config.confidence_threshold
            and not self._candidate_is_resolved(item)
            and (
                "origin=coverage" not in item.source
                or self.state.independent_detail_support(
                    item.interval,
                    self.config.focus_overlap_threshold,
                )
                >= 2
            )
        ]
        return max(candidates, key=lambda item: item.confidence) if candidates else None

    def _coverage_goal(self) -> dict[str, Any]:
        avoid = [target for strategy, target in self.goal_targets if strategy != "overview"]
        target = self.state.next_unobserved_interval(
            self.state.analysis_window_points,
            avoid=avoid,
        )
        if target is None:
            return {
                "thought": "All available detail focus regions have already been investigated.",
                "goal": "Finish because no novel detail interval remains.",
                "strategy": "finish",
                "target_intervals": [[0, self.state.series_length - 1]],
                "reference_index": -1,
                "done": True,
            }
        self.state.exploration_targets += 1
        return {
            "thought": "Coverage guard: move to the largest unobserved gap instead of revisiting prior evidence.",
            "goal": "Inspect a novel detail window in the largest remaining coverage gap.",
            "strategy": "coverage",
            "target_intervals": [target.to_list()],
            "reference_index": -1,
            "done": False,
        }

    def _fallback(self, step: int) -> dict[str, Any]:
        if not self.state.overview and not self.state.observed:
            target = Interval(0, self.state.series_length - 1)
            strategy = "overview"
            goal_text = "Inspect a global overview and characterize the dominant pattern and deviations."
        elif candidate := self._unresolved_overview_candidate():
            target = _window_around(candidate.interval, self.state.analysis_window_points, self.state.series_length)
            strategy = "zoom"
            goal_text = "Refine the strongest coarse overview candidate in a detail-scale window."
        elif candidate := self._strongest_detail_candidate():
            target = _window_around(candidate.interval, self.state.analysis_window_points, self.state.series_length)
            strategy = "compare"
            goal_text = "Compare the strongest detailed candidate with the best-matching learned reference."
        elif self.state.coverage < self.config.target_coverage:
            return self._coverage_goal()
        else:
            target = Interval(0, self.state.series_length - 1)
            strategy = "finish"
            goal_text = "Finish because the Planner has no unresolved visual evidence to investigate."
        return {
            "thought": "Deterministic safe fallback plan.",
            "goal": goal_text,
            "strategy": strategy,
            "target_intervals": [target.to_list()],
            "reference_index": -1,
            "done": strategy == "finish",
        }

    def _required_goal(self) -> dict[str, Any] | None:
        if not self.state.overview and not self.state.observed:
            return self._fallback(0)
        if candidate := self._unresolved_overview_candidate():
            zoom = _window_around(candidate.interval, self.state.analysis_window_points, self.state.series_length)
            return {
                "thought": "Resolution guard: refine an unresolved overview candidate before any Mark.",
                "goal": "Inspect a detail-scale window around the strongest unresolved overview candidate.",
                "strategy": "zoom",
                "target_intervals": [zoom.to_list()],
                "reference_index": -1,
                "done": False,
            }
        if self.config.mode == "optimized" and (candidate := self._strongest_compare_candidate()):
            return {
                "thought": "Candidate queue: accept a distinct event already supported by a stored Compare.",
                "goal": "Mark the strongest unhandled Compare-supported event without another visual call.",
                "strategy": "mark_candidate",
                "target_intervals": [candidate.interval.to_list()],
                "candidate": candidate.to_dict(),
                "reference_index": -1,
                "done": False,
            }
        if self.config.mode == "optimized" and (candidate := self._strongest_detail_candidate()):
            zoom = _window_around(candidate.interval, self.state.analysis_window_points, self.state.series_length)
            candidate_origin = "coverage" if "origin=coverage" in candidate.source else "detail"
            return {
                "thought": "Reference guard: compare one unresolved Plot candidate against normal memory.",
                "goal": "Compare the unresolved detail candidate with the best-matching learned reference once.",
                "strategy": "compare",
                "candidate_origin": candidate_origin,
                "target_intervals": [zoom.to_list()],
                "reference_index": -1,
                "done": False,
            }
        return None

    def _target_is_redundant(self, strategy: str, target: Interval) -> bool:
        if strategy == "mark_candidate":
            return False
        if strategy in {"zoom", "coverage"}:
            covered = sum(
                target.intersection_length(observed)
                for observed in merge_intervals(self.state.observed)
            )
            if covered / target.length >= 0.8:
                return True
        return any(
            previous_strategy == strategy
            and interval_overlap_fraction(target, previous) >= self.config.focus_overlap_threshold
            for previous_strategy, previous in self.goal_targets
        )

    def next_goal(self, step: int) -> dict[str, Any]:
        goal = self._required_goal()
        if goal is not None:
            self.trace.log("planner_state_guard", {"step": step, "goal": goal})
        else:
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

        target = _target_from_goal(goal, self.state.series_length)
        strategy = str(goal.get("strategy", "overview")).lower()
        premature_finish = bool(goal.get("done", False)) and self.state.coverage < self.config.target_coverage
        if premature_finish or self._target_is_redundant(strategy, target):
            original = {"strategy": strategy, "target": target.to_list(), "done": bool(goal.get("done", False))}
            goal = self._coverage_goal()
            target = _target_from_goal(goal, self.state.series_length)
            strategy = str(goal.get("strategy", "finish")).lower()
            self.state.novelty_redirects += 1
            self.trace.log(
                "planner_novelty_redirect",
                {"step": step, "rejected": original, "replacement": goal},
            )
        self.goal_targets.append((strategy, target))
        goal["target_intervals"] = [target.to_list()]
        try:
            goal["reference_index"] = int(goal.get("reference_index", -1))
        except (TypeError, ValueError):
            goal["reference_index"] = -1
        goal["done"] = bool(goal.get("done", False)) and bool(self.state.overview or self.state.observed)
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
                args["reference_index"] = int(goal.get("reference_index", -1))
            return {"thought": "Acquire visual evidence for the target interval.", "tool": tool, "args": args}
        for observation in reversed(observations):
            anomalies = observation.get("anomalies", []) if isinstance(observation, dict) else []
            if anomalies:
                item = max(anomalies, key=lambda value: float(value.get("confidence", 0.0)))
                interval = Interval(int(item["start"]), int(item["end"]))
                overview_needs_detail = not any(
                    observed.start <= interval.start and interval.end <= observed.end
                    for observed in self.state.observed
                )
                if (
                    observation.get("tool") == "Plot"
                    and observation.get("resolution") == "overview"
                    and overview_needs_detail
                ):
                    detail = _window_around(interval, self.state.analysis_window_points, self.state.series_length)
                    return {
                        "thought": "Refine the coarse overview candidate at detail resolution before marking.",
                        "tool": "Plot",
                        "args": {"start": detail.start, "end": detail.end},
                    }
                if observation.get("tool") == "Plot" and not any(
                    item_observation.get("tool") == "Compare"
                    for item_observation in observations
                    if isinstance(item_observation, dict)
                ):
                    return {
                        "thought": "Ground the detailed candidate against the best-matching reference.",
                        "tool": "Compare",
                        "args": {
                            "start": interval.start,
                            "end": interval.end,
                            "reference_index": -1,
                        },
                    }
                existing = [mark for mark in self.state.marks if mark.interval.overlaps(interval)]
                candidate_confidence = float(item.get("confidence", 0.0))
                candidate_is_compare = "compare" in str(item.get("source", ""))
                existing_has_compare = any("compare" in mark.source for mark in existing)
                if (
                    not existing
                    or max(mark.confidence for mark in existing) < candidate_confidence
                    or (candidate_is_compare and not existing_has_compare)
                ):
                    return {
                        "thought": "Register or revise the strongest detail-supported candidate.",
                        "tool": "Mark",
                        "args": item,
                    }
        return {"thought": "No further evidence-based action is needed.", "tool": "Finish", "args": {"reason": "goal complete"}}

    def run(self, goal: dict[str, Any]) -> dict[str, Any]:
        observations: list[dict[str, Any]] = []
        actions: list[dict[str, Any]] = []
        seen: dict[str, int] = {}
        seen_visuals: list[tuple[str, Interval]] = []
        self.state.active_strategy = str(goal.get("strategy", "")).lower()
        if self.state.active_strategy == "mark_candidate":
            raw_candidate = goal.get("candidate", {})
            candidate = raw_candidate if isinstance(raw_candidate, dict) else {}
            observation = self.environment.execute("Mark", candidate)
            actions.append(
                {
                    "thought": "Reuse stored Compare evidence for this distinct queued event.",
                    "tool": "Mark",
                    "args": candidate,
                }
            )
            observations.append(observation)
            if observation.get("status") != "ok":
                for item in self.state.analyzer_candidates:
                    if (
                        item.interval.start == candidate.get("start")
                        and item.interval.end == candidate.get("end")
                        and item.source == candidate.get("source")
                    ):
                        self.state.dismiss_candidate(item)
                        break
            task_summary = self.summarizer.summarize(goal, actions, observations)
            summary = {
                "goal": str(goal.get("goal", "")),
                "strategy": self.state.active_strategy,
                "summary": task_summary,
                "actions": ["Mark"],
                "new_observations": 1,
                "coverage_after": round(self.state.coverage, 6),
                "mark_count_after": len(self.state.marks),
                "last_observation": observation,
            }
            self.trace.log("candidate_queue_mark", summary)
            self.trace.log("subtask_summary", summary)
            return summary
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
                        "reference_index": int(goal.get("reference_index", -1)),
                    },
                }
                tool = "Compare"
                args = action["args"]
            signature = json.dumps({"tool": tool, "args": args}, sort_keys=True, ensure_ascii=False)
            seen[signature] = seen.get(signature, 0) + 1
            near_repeat = False
            action_interval: Interval | None = None
            if tool in {"Plot", "Compare"}:
                try:
                    start, end = sorted((int(args.get("start", 0)), int(args.get("end", 0))))
                    action_interval = Interval(max(0, start), max(0, end)).clamp(self.state.series_length)
                    near_repeat = any(
                        previous_tool == tool
                        and interval_overlap_fraction(action_interval, previous) >= self.config.focus_overlap_threshold
                        for previous_tool, previous in seen_visuals
                    )
                except (TypeError, ValueError):
                    action_interval = None
            if seen[signature] > 1 or near_repeat:
                self.trace.log(
                    "executor_repetition",
                    {"tool_step": tool_step, "action": action, "near_repeat": near_repeat},
                )
                action = self._fallback_action(goal, observations)
                tool = action["tool"]
                args = action["args"]
                fallback_signature = json.dumps(
                    {"tool": tool, "args": args},
                    sort_keys=True,
                    ensure_ascii=False,
                )
                fallback_near_repeat = False
                fallback_interval: Interval | None = None
                if tool in {"Plot", "Compare"}:
                    try:
                        start, end = sorted((int(args.get("start", 0)), int(args.get("end", 0))))
                        fallback_interval = Interval(max(0, start), max(0, end)).clamp(
                            self.state.series_length
                        )
                        fallback_near_repeat = any(
                            previous_tool == tool
                            and interval_overlap_fraction(fallback_interval, previous)
                            >= self.config.focus_overlap_threshold
                            for previous_tool, previous in seen_visuals
                        )
                    except (TypeError, ValueError):
                        fallback_interval = None
                if seen.get(fallback_signature, 0) > 0 or fallback_near_repeat:
                    action = {
                        "thought": "Novelty guard: the fallback would revisit the same focus.",
                        "tool": "Finish",
                        "args": {"reason": "no non-redundant evidence action remains"},
                    }
                    tool = "Finish"
                    args = action["args"]
                else:
                    seen[fallback_signature] = 1
                    if fallback_interval is not None:
                        seen_visuals.append((tool, fallback_interval))
            elif action_interval is not None:
                seen_visuals.append((tool, action_interval))

            coverage_origin = (
                str(goal.get("strategy", "")).lower() == "coverage"
                or str(goal.get("candidate_origin", "")).lower() == "coverage"
            )
            if tool == "Mark" and coverage_origin:
                try:
                    left, right = sorted(
                        (int(args.get("start", 0)), int(args.get("end", args.get("start", 0))))
                    )
                    mark_interval = Interval(max(0, left), max(0, right)).clamp(
                        self.state.series_length
                    )
                except (TypeError, ValueError):
                    mark_interval = Interval(0, 0)
                support_count = self.state.independent_detail_support(
                    mark_interval,
                    self.config.focus_overlap_threshold,
                )
                if support_count < 2:
                    self.trace.log(
                        "executor_coverage_mark_rejected",
                        {
                            "tool_step": tool_step,
                            "mark_interval": mark_interval.to_list(),
                            "independent_visual_support": support_count,
                            "required_visual_support": 2,
                        },
                    )
                    action = {
                        "thought": "Coverage event has only one visual source; leave it unmarked and move on.",
                        "tool": "Finish",
                        "args": {"reason": "coverage event lacks a second independent visual observation"},
                    }
                    tool = "Finish"
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
            if tool == "Mark" and observation.get("status") == "ok":
                self.trace.log(
                    "executor_mark_resolved_event",
                    {"tool_step": tool_step, "mark": observation.get("mark", {})},
                )
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

    def _learn_reference_memory(
        self,
        dataset: str,
        sample_id: str,
        reference_series: np.ndarray,
        reference_intervals: list[Interval],
        reference_cycles: list[float],
        reference_source: str,
        context: dict[str, Any],
    ) -> dict[str, Any]:
        numeric_catalog = build_reference_catalog(reference_series, reference_intervals, reference_cycles)
        default_reliability = 0.95 if reference_source == "train" else 0.55
        fallback_profiles = [
            {
                **item,
                "description": "Numeric fallback profile; visual reference learning was unavailable.",
                "reliability": default_reliability,
                "possible_contamination": reference_source != "train",
                "distinctive_features": [],
            }
            for item in numeric_catalog
        ]
        fallback = {
            "status": "numeric-fallback",
            "source": reference_source,
            "normal_pattern": (
                "Use scale-aware consensus from the numeric profiles. Compare queries only with references carrying "
                "the same window_cycles. Training references are designated normal; test-unsupervised references are "
                "fallible candidates and must not be treated as ground truth."
            ),
            "reference_profiles": fallback_profiles,
            "selection_guidance": (
                "Probe 1P, 3P and 5P queries against same-scale references; then choose the scale with sufficient "
                "context and the clearest reliable anomaly evidence."
            ),
            "uncertainties": ["No visual reference-learning response is available."],
        }
        if not reference_intervals or not self.config.enable_reference_learning:
            fallback["status"] = "disabled" if reference_intervals else "no-references"
            return fallback

        images: list[Path] = []
        for index, interval in enumerate(reference_intervals):
            cycle_label = f"{reference_cycles[index]:g}P"
            path = self.run_dir / "plots" / f"reference_{index:02d}_{interval.start}_{interval.end}.png"
            render_interval(
                reference_series,
                interval,
                path,
                f"Reference candidate {index} - {cycle_label} ({reference_source})",
                self.config.plot_max_points,
            )
            images.append(path)
        try:
            response = self.client.call_json(
                REFERENCE_SYSTEM,
                reference_prompt(dataset, sample_id, reference_source, numeric_catalog, context),
                images,
                max_tokens=1600,
            )
            raw_profiles = response.content.get("reference_profiles", [])
            by_index: dict[int, dict[str, Any]] = {}
            for item in raw_profiles if isinstance(raw_profiles, list) else []:
                if not isinstance(item, dict):
                    continue
                try:
                    index = int(item.get("reference_index"))
                except (TypeError, ValueError):
                    continue
                if 0 <= index < len(numeric_catalog):
                    by_index[index] = item
            profiles: list[dict[str, Any]] = []
            for base in numeric_catalog:
                index = int(base["reference_index"])
                visual = by_index.get(index, {})
                try:
                    reliability = float(visual.get("reliability", default_reliability))
                except (TypeError, ValueError):
                    reliability = default_reliability
                if not np.isfinite(reliability):
                    reliability = default_reliability
                profiles.append(
                    {
                        **base,
                        "description": str(visual.get("description", ""))[:1600],
                        "reliability": min(1.0, max(0.0, reliability)),
                        "possible_contamination": bool(
                            visual.get("possible_contamination", reference_source != "train")
                        ),
                        "distinctive_features": [str(value)[:300] for value in visual.get("distinctive_features", [])[:8]]
                        if isinstance(visual.get("distinctive_features", []), list)
                        else [],
                    }
                )
            memory = {
                "status": "visual-learned",
                "source": reference_source,
                "normal_pattern": str(response.content.get("normal_pattern", ""))[:5000],
                "reference_profiles": profiles,
                "selection_guidance": str(response.content.get("selection_guidance", ""))[:2000],
                "uncertainties": [str(value)[:500] for value in response.content.get("uncertainties", [])[:8]]
                if isinstance(response.content.get("uncertainties", []), list)
                else [],
            }
            self.trace.log(
                "reference_learning_response",
                {
                    "catalog": numeric_catalog,
                    "images": [str(path) for path in images],
                    "memory": memory,
                    **_model_record(response),
                },
            )
            return memory
        except Exception as exc:
            self.trace.log(
                "reference_learning_error",
                {"error": str(exc), "catalog": numeric_catalog, "used_numeric_fallback": True},
            )
            return fallback

    @staticmethod
    def _weighted_median(values: list[tuple[int, float]]) -> int:
        ordered = sorted(values, key=lambda item: item[0])
        cutoff = sum(weight for _, weight in ordered) / 2.0
        cumulative = 0.0
        for value, weight in ordered:
            cumulative += weight
            if cumulative >= cutoff:
                return int(value)
        return int(ordered[-1][0])

    def _reflection_candidates(self, state: AgentState) -> list[dict[str, Any]]:
        """Build robust reflection candidates from every detailed observation."""

        if self.config.mode != "optimized":
            return []
        event_gap = max(self.config.merge_gap, self.config.min_plot_points // 2)

        def related(left: AnomalyMark, right: AnomalyMark) -> bool:
            if left.interval.overlaps(right.interval):
                return True
            gap = max(
                0,
                max(left.interval.start, right.interval.start)
                - min(left.interval.end, right.interval.end)
                - 1,
            )
            return left.anomaly_type == right.anomaly_type and gap <= event_gap

        eligible = [*state.marks]
        eligible.extend(
            item
            for item in state.analyzer_candidates
            if item.source.startswith("detail:")
            and item.confidence >= self.config.confidence_threshold
        )

        # Mark actions copy analyzer candidates with a different source.  Count an
        # identical interval/type only once so an action is not mistaken for a
        # second independent visual observation.
        unique: dict[tuple[int, int, str], AnomalyMark] = {}
        for candidate in eligible:
            signature = (
                candidate.interval.start,
                candidate.interval.end,
                candidate.anomaly_type,
            )
            previous = unique.get(signature)
            if previous is None or candidate.confidence > previous.confidence:
                unique[signature] = candidate

        # Two detail views often identify different fragments of one visual
        # event.  Join fragments separated by at most half the minimum detail
        # plot width, while keeping distant events (for example UCR/135's two
        # candidates 93 points apart) separate.
        clusters: list[list[AnomalyMark]] = []
        for candidate in sorted(unique.values(), key=lambda item: item.interval.start):
            matches = [
                index
                for index, cluster in enumerate(clusters)
                if any(related(candidate, member) for member in cluster)
            ]
            if not matches:
                clusters.append([candidate])
                continue
            first = matches[0]
            clusters[first].append(candidate)
            for index in reversed(matches[1:]):
                clusters[first].extend(clusters.pop(index))

        result: list[dict[str, Any]] = []
        for cluster in clusters:
            by_observation: dict[str, AnomalyMark] = {}
            for item in cluster:
                observation = state.observation_key(item.source)
                previous = by_observation.get(observation)
                if previous is None or item.confidence > previous.confidence:
                    by_observation[observation] = item
            evidence_items = list(by_observation.values())
            support_count = len(evidence_items)
            coverage_only = all("origin=coverage" in item.source for item in cluster)
            if coverage_only and support_count < 2:
                self.trace.log(
                    "reflection_single_source_coverage_rejected",
                    {
                        "support_count": support_count,
                        "intervals": [item.interval.to_list() for item in cluster],
                        "sources": sorted(by_observation),
                    },
                )
                continue
            if support_count == 2:
                # With only two observations there is no robust median.  Their
                # union preserves complementary fragments of the same nearby
                # event; three or more proposals use the outlier-resistant
                # weighted median below.
                start = min(item.interval.start for item in evidence_items)
                end = max(item.interval.end for item in evidence_items)
            else:
                start = self._weighted_median(
                    [(item.interval.start, max(item.confidence, 1e-6)) for item in evidence_items]
                )
                end = self._weighted_median(
                    [(item.interval.end, max(item.confidence, 1e-6)) for item in evidence_items]
                )
            start, end = sorted((start, end))
            type_scores: dict[str, float] = {}
            for item in evidence_items:
                type_scores[item.anomaly_type] = (
                    type_scores.get(item.anomaly_type, 0.0) + item.confidence
                )
            anomaly_type = max(type_scores, key=type_scores.get)
            representative = max(
                evidence_items,
                key=lambda item: (
                    item.anomaly_type == anomaly_type,
                    item.confidence,
                    item.interval.length,
                ),
            )
            intervals = sorted({(item.interval.start, item.interval.end) for item in cluster})
            interval_text = ", ".join(f"[{left},{right}]" for left, right in intervals)
            evidence = (
                f"Consensus from {support_count} distinct detail proposals ({interval_text}); "
                f"weighted boundary [{start},{end}], majority type {anomaly_type}. "
                f"Representative evidence: {representative.evidence}"
            )[:1200]
            result.append(
                {
                    "mark": AnomalyMark(
                        interval=Interval(start, end),
                        confidence=max(item.confidence for item in evidence_items),
                        anomaly_type=anomaly_type,
                        evidence=evidence,
                        source=f"detail-consensus:{support_count}",
                    ),
                    "support_count": support_count,
                    "coverage_only": coverage_only,
                    "event_gap": event_gap,
                    "supporting_intervals": [list(item) for item in intervals],
                    "supporting_sources": sorted(by_observation),
                }
            )
        result.sort(
            key=lambda item: (
                item["mark"].confidence,
                item["support_count"],
            ),
            reverse=True,
        )
        return result

    def _reflect(
        self,
        series: np.ndarray,
        state: AgentState,
        reference_series: np.ndarray | None = None,
    ) -> None:
        if not self.config.enable_reflection:
            return
        candidate_groups = self._reflection_candidates(state)
        numeric_reference = series if reference_series is None else reference_series
        retained_groups: list[dict[str, Any]] = []
        rejected_coverage: list[Interval] = []
        for group in candidate_groups:
            candidate = group["mark"]
            if not group.get("coverage_only", False):
                retained_groups.append(group)
                continue
            _, numeric_evidence = self._refine_roughness_interval(
                series,
                numeric_reference,
                candidate.interval,
                state.analysis_window_points,
                candidate.anomaly_type,
            )
            if (
                candidate.confidence >= self.config.coverage_confidence_threshold
                or numeric_evidence is not None
            ):
                group["coverage_numeric_evidence"] = numeric_evidence
                retained_groups.append(group)
                continue
            rejected_coverage.append(candidate.interval)
            self.trace.log(
                "reflection_weak_coverage_rejected",
                {
                    "candidate": candidate.to_dict(),
                    "confidence_threshold": self.config.coverage_confidence_threshold,
                    "numeric_evidence": numeric_evidence,
                    "supporting_sources": group.get("supporting_sources", []),
                },
            )
        candidate_groups = retained_groups
        if rejected_coverage:
            state.marks = [
                mark
                for mark in state.marks
                if not (
                    "origin=coverage" in mark.source
                    and any(mark.interval.overlaps(interval) for interval in rejected_coverage)
                )
            ]
        if not candidate_groups:
            self.trace.log("reflection_skipped", {"reason": "no candidates"})
            return
        consensus_fallback = [
            group["mark"]
            for group in candidate_groups
            if group["support_count"] >= 2
        ][: self.config.max_marks]
        images: list[Path] = []
        metadata: list[dict[str, Any]] = []
        for index, group in enumerate(candidate_groups):
            candidate = group["mark"]
            padding = max(128, candidate.interval.length * 4)
            view = Interval(
                max(0, candidate.interval.start - padding),
                min(len(series) - 1, candidate.interval.end + padding),
            )
            path = self.run_dir / "plots" / f"reflection_{index:02d}_{view.start}_{view.end}.png"
            render_interval(
                series,
                view,
                path,
                f"Reflection candidate {index + 1}: proposed [{candidate.interval.start}, {candidate.interval.end}]",
                self.config.plot_max_points,
                highlight=candidate.interval,
            )
            images.append(path)
            item = candidate.to_dict()
            item["candidate_id"] = index
            item["visible_range"] = view.to_list()
            item["candidate_within_visible_range"] = (
                view.start <= candidate.interval.start
                and candidate.interval.end <= view.end
            )
            item["support_count"] = group["support_count"]
            item["coverage_only"] = group["coverage_only"]
            item["coverage_numeric_evidence"] = group.get("coverage_numeric_evidence")
            item["event_gap"] = group["event_gap"]
            item["supporting_intervals"] = group["supporting_intervals"]
            item["supporting_sources"] = group["supporting_sources"]
            metadata.append(item)
        self.trace.log("reflection_candidate_consensus", {"candidates": metadata})
        try:
            response = self.client.call_json(
                REFLECTION_SYSTEM,
                reflection_prompt(metadata, state.reference_memory),
                images,
                max_tokens=1200,
            )
            self.trace.log("reflection_response", {"candidates": metadata, **_model_record(response)})
            verified_raw = response.content.get("verified_anomalies", [])
            verified: list[AnomalyMark] = []
            visible = {int(item["candidate_id"]): Interval(*item["visible_range"]) for item in metadata}
            for item in verified_raw if isinstance(verified_raw, list) else []:
                if not isinstance(item, dict):
                    continue
                try:
                    candidate_id = int(item.get("candidate_id"))
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
                if candidate_id not in visible or not interval.overlaps(visible[candidate_id]):
                    continue
                kind = str(item.get("anomaly_type", "unknown")).strip().lower()
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
            if verified:
                state.marks = verified[: self.config.max_marks]
            else:
                state.marks = consensus_fallback
                self.trace.log(
                    "reflection_empty_consensus_fallback",
                    {
                        "kept_count": len(consensus_fallback),
                        "minimum_support": 2,
                        "kept_marks": [item.to_dict() for item in consensus_fallback],
                    },
                )
        except Exception as exc:
            if consensus_fallback:
                state.marks = consensus_fallback
            self.trace.log(
                "reflection_error",
                {
                    "error": str(exc),
                    "kept_consensus_marks": len(consensus_fallback),
                    "kept_original_marks": not bool(consensus_fallback),
                },
            )

    @staticmethod
    def _refine_roughness_interval(
        series: np.ndarray,
        reference_series: np.ndarray,
        candidate: Interval,
        analysis_window_points: int,
        anomaly_type: str,
    ) -> tuple[Interval, dict[str, Any] | None]:
        """Correct visual coordinate drift using label-free roughness evidence.

        Shape/frequency/point events usually create unusually large adjacent
        differences.  The threshold comes only from the normal-reference
        series; the model candidate merely limits the search neighborhood.
        """

        if anomaly_type not in {"shapelet", "frequency", "point"}:
            return candidate, None
        values = np.asarray(series, dtype=float).reshape(-1)
        reference = np.asarray(reference_series, dtype=float).reshape(-1)
        if len(values) < 2 or len(reference) < 2:
            return candidate, None
        differences = np.abs(np.diff(values, prepend=values[0]))
        reference_differences = np.abs(np.diff(reference, prepend=reference[0]))
        median = float(np.median(reference_differences))
        mad = float(np.median(np.abs(reference_differences - median)))
        threshold = max(
            median + 8.0 * 1.4826 * mad,
            float(np.quantile(reference_differences, 0.995)),
        )
        if not np.isfinite(threshold):
            return candidate, None

        window = max(1, int(analysis_window_points))
        search_padding = max(candidate.length, window // 10)
        search_start = max(0, candidate.start - search_padding)
        search_end = min(len(values) - 1, candidate.end + search_padding)
        high = np.flatnonzero(differences[search_start : search_end + 1] > threshold)
        if not len(high):
            return candidate, None
        high = high + search_start
        join_gap = max(3, window // 20)
        clusters: list[list[int]] = [[int(high[0])]]
        for raw_index in high[1:]:
            index = int(raw_index)
            if index - clusters[-1][-1] <= join_gap:
                clusters[-1].append(index)
            else:
                clusters.append([index])

        def distance(cluster: list[int]) -> int:
            if cluster[-1] < candidate.start:
                return candidate.start - cluster[-1]
            if candidate.end < cluster[0]:
                return cluster[0] - candidate.end
            return 0

        associated = [cluster for cluster in clusters if distance(cluster) <= join_gap]
        minimum_spikes = 1 if anomaly_type == "point" else 2
        associated = [cluster for cluster in associated if len(cluster) >= minimum_spikes]
        if not associated:
            return candidate, None
        selected = max(
            associated,
            key=lambda cluster: (
                float(np.sum(differences[cluster] - threshold)),
                -distance(cluster),
                len(cluster),
            ),
        )
        halo = max(1, window // 250)
        refined = Interval(
            max(0, selected[0] - halo),
            min(len(values) - 1, selected[-1] + halo),
        )
        return refined, {
            "method": "reference_roughness",
            "model_interval": candidate.to_list(),
            "numeric_interval": refined.to_list(),
            "threshold": round(threshold, 6),
            "spike_count": len(selected),
        }

    def _final_predictions(
        self,
        state: AgentState,
        series: np.ndarray,
        reference_series: np.ndarray,
    ) -> list[dict[str, Any]]:
        groups = merge_intervals([mark.interval for mark in state.marks], gap=self.config.merge_gap)
        predictions: list[dict[str, Any]] = []
        for group in groups:
            members = [mark for mark in state.marks if mark.interval.overlaps(group)]
            strongest = max(members, key=lambda item: item.confidence)
            refined, refinement = self._refine_roughness_interval(
                series,
                reference_series,
                group,
                state.analysis_window_points,
                strongest.anomaly_type,
            )
            item = {
                "start": refined.start,
                "end": refined.end,
                "confidence": strongest.confidence,
                "anomaly_type": strongest.anomaly_type,
                "evidence": strongest.evidence,
                "source": strongest.source,
            }
            if refinement is not None:
                item["boundary_refinement"] = refinement
            predictions.append(item)
        return predictions

    def run(self, sample: TimeSeriesSample) -> dict[str, Any]:
        series, normalization = robust_standardize(sample.values)
        reference_cycles = [float(value) for value in self.config.reference_cycle_plan]
        reference_windows, estimated_period = multiscale_window_points(
            sample,
            reference_cycles,
            min_points=self.config.reference_min_points,
            max_points=self.config.reference_max_points,
        )
        adaptive_cycles = [float(value) for value in self.config.adaptive_compare_cycles]
        adaptive_windows, _ = multiscale_window_points(
            sample,
            adaptive_cycles,
            min_points=self.config.reference_min_points,
            max_points=self.config.reference_max_points,
        )
        analysis_windows = {
            f"{cycles:g}": points for cycles, points in zip(adaptive_cycles, adaptive_windows)
        }
        default_key = f"{float(self.config.reference_window_cycles):g}"
        window_points = analysis_windows.get(default_key)
        if window_points is None:
            default_windows, _ = multiscale_window_points(
                sample,
                [self.config.reference_window_cycles],
                min_points=self.config.reference_min_points,
                max_points=self.config.reference_max_points,
            )
            window_points = default_windows[0]
            analysis_windows[default_key] = window_points
        reference_series, reference_intervals, reference_source = select_reference_intervals(
            sample,
            count=len(reference_cycles),
            window_points=reference_windows,
        )
        dataset_context = {
            "metadata": sample.metadata,
            "reference_source": reference_source,
            "normalization": "median/MAD robust standardization",
            "labels_available_to_agent": False,
            "label_derived_metadata_available_to_agent": False,
            "analysis_window_points": window_points,
            "analysis_windows": analysis_windows,
            "reference_cycle_plan": reference_cycles,
            "estimated_period": estimated_period,
        }
        reference_memory = self._learn_reference_memory(
            sample.dataset,
            sample.sample_id,
            reference_series,
            reference_intervals,
            reference_cycles,
            reference_source,
            dataset_context,
        )
        state = AgentState(
            dataset=sample.dataset,
            sample_id=sample.sample_id,
            series_length=sample.length,
            dataset_context=dataset_context,
            analysis_window_points=window_points,
            event_guard_points=self.config.event_guard_points,
            analysis_windows=analysis_windows,
            estimated_period=estimated_period,
            reference_memory=reference_memory,
        )
        environment = ToolEnvironment(
            series=series,
            reference_series=reference_series,
            reference_intervals=reference_intervals,
            reference_cycles=reference_cycles,
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
                "reference_cycles": reference_cycles,
                "reference_window_points": reference_windows,
                "reference_memory": reference_memory,
                "analysis_window_points": window_points,
                "analysis_windows": analysis_windows,
                "estimated_period": estimated_period,
            },
        )

        for planner_step in range(self.config.max_planner_steps):
            goal = planner.next_goal(planner_step)
            if goal.get("done"):
                self.trace.log("planner_finish", {"step": planner_step, "goal": goal})
                break
            summary = executor.run(goal)
            state.task_history.append(summary)

        self._reflect(series, state, reference_series)
        predictions = self._final_predictions(state, series, reference_series)
        attention_summary = state.attention_summary()
        state.finished = True
        result = {
            "schema_version": 2,
            "dataset": sample.dataset,
            "sample_id": sample.sample_id,
            "mode": self.config.mode,
            "model": self.client.config.model,
            "series_length": sample.length,
            "normalization": normalization,
            "reference_source": reference_source,
            "reference_intervals": [item.to_list() for item in reference_intervals],
            "reference_cycles": reference_cycles,
            "reference_window_points": reference_windows,
            "reference_memory": reference_memory,
            "analysis_window_points": window_points,
            "analysis_windows": analysis_windows,
            "estimated_period": estimated_period,
            # Kept for schema-v1 consumers; in schema v2 this means detail
            # coverage only and never counts a compressed global overview.
            "coverage": state.coverage,
            "detail_coverage": state.coverage,
            "overview_coverage": state.overview_coverage,
            "predictions": predictions,
            "overview_intervals": [item.to_list() for item in state.overview],
            "observed_intervals": [item.to_list() for item in state.observed],
            "attention_summary": attention_summary,
            "subtask_history": state.task_history,
            "label_policy": "ground-truth labels were not loaded during inference",
        }
        save_json(self.run_dir / "predictions.json", result)
        self.trace.log(
            "run_complete",
            {
                "prediction_count": len(predictions),
                "detail_coverage": state.coverage,
                "overview_coverage": state.overview_coverage,
                "attention_summary": attention_summary,
            },
        )
        return result
