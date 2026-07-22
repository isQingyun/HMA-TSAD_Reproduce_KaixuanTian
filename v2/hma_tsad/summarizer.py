from __future__ import annotations

import json
from typing import Any

from .client import DashScopeClient
from .prompts import TASK_SUMMARIZER_SYSTEM, task_summarizer_prompt
from .state import TraceLogger


class TaskSummarizer:
    """Paper-faithful M_summary: condense one Executor trajectory for the Planner."""

    def __init__(
        self,
        client: DashScopeClient,
        trace: TraceLogger,
        max_tokens: int = 500,
    ) -> None:
        self.client = client
        self.trace = trace
        self.max_tokens = max_tokens

    @staticmethod
    def _trajectory(
        actions: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            {
                "step": index,
                "thought": str(action.get("thought", "")),
                "action": {
                    "tool": str(action.get("tool", "")),
                    "args": action.get("args", {}),
                },
                "observation": observations[index] if index < len(observations) else {},
            }
            for index, action in enumerate(actions)
        ]

    @staticmethod
    def _fallback_summary(
        goal: dict[str, Any],
        trajectory: list[dict[str, Any]],
    ) -> str:
        tools = [str(item.get("action", {}).get("tool", "")) for item in trajectory]
        final_observation = trajectory[-1].get("observation", {}) if trajectory else {}
        compact_observation = json.dumps(
            final_observation,
            ensure_ascii=False,
            separators=(",", ":"),
        )[:1200]
        return (
            f"Sub-task {str(goal.get('goal', ''))!r} executed tools {tools} and produced "
            f"{len(trajectory)} observations. Final observation: {compact_observation or '{}'}"
        )[:2000]

    def summarize(
        self,
        goal: dict[str, Any],
        actions: list[dict[str, Any]],
        observations: list[dict[str, Any]],
    ) -> str:
        trajectory = self._trajectory(actions, observations)
        self.trace.log(
            "task_summarizer_request",
            {"goal": goal, "trajectory": trajectory},
        )
        try:
            response = self.client.call_json(
                TASK_SUMMARIZER_SYSTEM,
                task_summarizer_prompt(goal, trajectory),
                max_tokens=self.max_tokens,
            )
            raw_summary = response.content.get("summary")
            if not isinstance(raw_summary, str) or not raw_summary.strip():
                raise ValueError("Task Summarizer response must contain a non-empty string field 'summary'")
            summary = raw_summary.strip()[:2000]
            self.trace.log(
                "task_summarizer_response",
                {
                    "goal": goal,
                    "trajectory_steps": len(trajectory),
                    "summary": summary,
                    "usage": response.usage,
                    "request_id": response.request_id,
                    "model": response.model,
                },
            )
            return summary
        except Exception as exc:
            summary = self._fallback_summary(goal, trajectory)
            self.trace.log(
                "task_summarizer_error",
                {
                    "goal": goal,
                    "trajectory_steps": len(trajectory),
                    "error": str(exc),
                    "fallback_summary": summary,
                },
            )
            return summary
