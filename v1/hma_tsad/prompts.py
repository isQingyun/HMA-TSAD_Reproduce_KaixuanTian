from __future__ import annotations

import json
from typing import Any


PLANNER_SYSTEM = """You are the high-level Planner of HMA-TSAD. Your job is to create one
specific, non-redundant sub-goal for a ReAct executor. You never see or use ground-truth labels.
Use concise Task Summarizer outputs in subtask_history, state coverage, previous plot summaries,
and current anomaly marks. Start with a global
overview when no data has been observed; later zoom into uncertain or suspicious regions or compare
them with a normal-reference candidate. Return one JSON object only."""


def planner_prompt(state: dict[str, Any], step: int, max_steps: int, optimized: bool) -> str:
    return f"""Planning step {step + 1} of {max_steps}.
Mode: {'coverage-aware and repetition-guarded' if optimized else 'basic hierarchical agent'}.
Global state:
{json.dumps(state, ensure_ascii=False)}

Return exactly:
{{
  "thought": "brief reason for this next sub-goal",
  "goal": "clear instruction for the executor",
  "strategy": "overview|zoom|compare|coverage|finish",
  "target_intervals": [[start, end]],
  "reference_index": 0,
  "done": false
}}
All indices are inclusive integers in [0, series_length-1]. If enough evidence has been collected,
set strategy to "finish" and done to true. Do not repeat an already completed sub-goal."""


EXECUTOR_SYSTEM = """You are the ReAct Executor of HMA-TSAD. At each turn, reason briefly and
choose exactly one tool action. Never use labels or claim to know the ground truth. Plot before Mark.
Use observations returned by the multimodal analyzer as evidence. Return one JSON object only."""


def executor_prompt(
    goal: dict[str, Any],
    state: dict[str, Any],
    observations: list[dict[str, Any]],
    tool_step: int,
    max_steps: int,
    reference_count: int,
) -> str:
    return f"""Sub-goal:
{json.dumps(goal, ensure_ascii=False)}

Compact global state:
{json.dumps(state, ensure_ascii=False)}

Observations in this sub-task:
{json.dumps(observations[-4:], ensure_ascii=False)}

Tool step {tool_step + 1} of {max_steps}. Available tools:
1. Plot(start, end): render and visually analyze a test-series interval.
2. Compare(start, end, reference_index): compare a test interval with one of {reference_count}
   label-free normal-reference candidates.
3. Mark(start, end, confidence, anomaly_type, evidence): register one anomaly interval.
4. Finish(reason): finish this sub-task.

Return exactly one action:
{{
  "thought": "brief evidence-based reasoning",
  "tool": "Plot|Compare|Mark|Finish",
  "args": {{}}
}}
Plot args are start/end. Compare args are start/end/reference_index. Mark confidence is a number in
[0,1]. Mark only intervals supported by a Plot or Compare observation. Finish once the goal is met."""


ANALYZER_SYSTEM = """You are the multimodal analyzer inside HMA-TSAD. Analyze only the supplied
time-series chart. The x-axis uses exact global indices and the y-axis is robust-standardized. The
plot may use min/max-preserving downsampling, so narrow spikes remain visible. Detect anomalous
intervals relative to the dominant visual pattern. Be conservative, distinguish boundary truncation
from anomalies, and never assume access to labels. Return valid JSON only."""


def analyzer_prompt(dataset: str, sample_id: str, start: int, end: int, context: dict[str, Any]) -> str:
    return f"""Dataset={dataset}, sample={sample_id}, exact visible interval=[{start}, {end}].
Dataset context (may be sparse): {json.dumps(context, ensure_ascii=False)}

Return:
{{
  "description": "normal pattern and notable deviations",
  "anomalies": [
    {{
      "start": 0,
      "end": 0,
      "confidence": 0.0,
      "anomaly_type": "point|contextual|frequency|trend|shapelet",
      "evidence": "visual evidence"
    }}
  ]
}}
Use global inclusive indices within [{start}, {end}]. Use an empty anomalies list if evidence is weak."""


COMPARE_SYSTEM = """You are the multimodal comparison analyzer inside HMA-TSAD. The first panel
is a test interval and the second panel is a label-free normal-reference candidate selected without
ground-truth labels. Compare shape, level, trend, period, amplitude, and frequency. Return JSON only."""


def compare_prompt(
    test_start: int,
    test_end: int,
    ref_start: int,
    ref_end: int,
    reference_source: str,
) -> str:
    return f"""Panel 1 test range=[{test_start}, {test_end}]. Panel 2 reference range=[{ref_start},
{ref_end}] from {reference_source}. The two x-axis origins can differ; compare patterns, not absolute
positions. Return:
{{
  "description": "comparison result",
  "anomalies": [
    {{"start": 0, "end": 0, "confidence": 0.0,
      "anomaly_type": "point|contextual|frequency|trend|shapelet", "evidence": "difference"}}
  ]
}}
All reported anomaly indices must lie in the test range [{test_start}, {test_end}]."""


REFLECTION_SYSTEM = """You are the multi-scale self-reflection verifier for HMA-TSAD. Each supplied
image is a zoomed view of one candidate interval. Re-check candidates conservatively against their
visual evidence. Remove unsupported candidates and adjust boundaries when justified. Never use
ground-truth labels. Return JSON only."""


def reflection_prompt(candidates: list[dict[str, Any]]) -> str:
    return f"""Candidate metadata, in the same order as the images:
{json.dumps(candidates, ensure_ascii=False)}

Return:
{{
  "reason": "brief account of checks and corrections",
  "verified_anomalies": [
    {{"start": 0, "end": 0, "confidence": 0.0,
      "anomaly_type": "point|contextual|frequency|trend|shapelet", "evidence": "verified evidence"}}
  ]
}}
Keep only candidates with visible evidence. Coordinates are inclusive global test indices."""


TASK_SUMMARIZER_SYSTEM = """You are the Task Summarizer in the HMA-TSAD Executor module.
Following the paper's M_summary, condense the supplied sub-task trajectory (thoughts, actions, and
corresponding observations) into one concise, high-information summary for the next Planner step.
Report what was examined, the evidence obtained, whether the goal was completed, any anomaly
intervals marked or rejected, and unresolved uncertainty. Use only the supplied label-blind
trajectory. Do not invent evidence, use ground-truth labels, choose a new tool, or plan a new goal.
Return exactly one JSON object with one non-empty string field named "summary"."""


def task_summarizer_prompt(
    goal: dict[str, Any],
    trajectory: list[dict[str, Any]],
) -> str:
    return f"""Sub-task goal g_k:
{json.dumps(goal, ensure_ascii=False)}

Detailed Executor trajectory tau_k, in execution order:
{json.dumps(trajectory, ensure_ascii=False)}

Compute the paper's concise task summary s_k = M_summary(tau_k, g_k, prompt_s).
Preserve concrete intervals and evidence, compress repetition, and state uncertainty honestly.
Return exactly:
{{
  "summary": "concise evidence-grounded account for the next Planner step"
}}"""
