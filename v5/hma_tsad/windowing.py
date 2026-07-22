from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from .intervals import Interval, labels_to_intervals, merge_intervals


@dataclass(frozen=True)
class SeriesWindow:
    """One inclusive slice of a global time series."""

    index: int
    interval: Interval

    @property
    def length(self) -> int:
        return self.interval.length

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "start": self.interval.start,
            "end": self.interval.end,
            "length": self.length,
        }


def plan_series_windows(
    series_length: int,
    window_size: int,
    overlap_ratio: float = 0.0,
) -> tuple[list[SeriesWindow], int]:
    """Cover a series with fixed-step windows and one optional short tail.

    At the default overlap ratio of zero the stride is exactly ``window_size``.
    The final window is shorter when the series length is not divisible by the
    window size; it is never shifted backwards to create an implicit overlap.
    """

    if series_length <= 0:
        raise ValueError("series_length must be positive")
    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if not 0.0 <= overlap_ratio < 1.0:
        raise ValueError("overlap_ratio must be in [0, 1)")
    size = min(int(window_size), int(series_length))
    stride = max(1, int(round(size * (1.0 - float(overlap_ratio)))))
    windows: list[SeriesWindow] = []
    start = 0
    while start < series_length:
        end = min(series_length - 1, start + size - 1)
        windows.append(SeriesWindow(len(windows), Interval(start, end)))
        if end == series_length - 1:
            break
        start += stride
    return windows, stride


def _local_interval(item: dict[str, Any], window: SeriesWindow) -> Interval | None:
    try:
        start = int(round(float(item["start"])))
        end = int(round(float(item.get("end", item["start"]))))
    except (KeyError, TypeError, ValueError):
        return None
    start, end = sorted((start, end))
    if end < 0 or start >= window.length:
        return None
    return Interval(max(0, start), max(0, end)).clamp(window.length)


def map_window_predictions(
    window: SeriesWindow,
    predictions: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Map local slice predictions to global coordinates without labels."""

    mapped: list[dict[str, Any]] = []
    for raw in predictions:
        if not isinstance(raw, dict):
            continue
        local = _local_interval(raw, window)
        if local is None:
            continue
        item = dict(raw)
        item["start"] = window.interval.start + local.start
        item["end"] = window.interval.start + local.end
        item["window_index"] = window.index
        item["local_interval"] = local.to_list()
        refinement = item.get("boundary_refinement")
        if isinstance(refinement, dict):
            mapped_refinement = dict(refinement)
            for key in ("model_interval", "numeric_interval"):
                raw_interval = mapped_refinement.get(key)
                if isinstance(raw_interval, list) and len(raw_interval) == 2:
                    try:
                        local_refined = Interval(
                            max(0, int(raw_interval[0])),
                            max(0, int(raw_interval[1])),
                        ).clamp(window.length)
                    except (TypeError, ValueError):
                        continue
                    mapped_refinement[key] = [
                        window.interval.start + local_refined.start,
                        window.interval.start + local_refined.end,
                    ]
            item["boundary_refinement"] = mapped_refinement
        mapped.append(item)
    return mapped


def map_local_intervals(
    window: SeriesWindow,
    raw_intervals: Sequence[Sequence[int]],
) -> list[Interval]:
    result: list[Interval] = []
    for raw in raw_intervals:
        if not isinstance(raw, (list, tuple)) or len(raw) != 2:
            continue
        try:
            local = Interval(max(0, int(raw[0])), max(0, int(raw[1]))).clamp(window.length)
        except (TypeError, ValueError):
            continue
        result.append(
            Interval(
                window.interval.start + local.start,
                window.interval.start + local.end,
            )
        )
    return result


def aggregate_window_predictions(
    series_length: int,
    windows_and_predictions: Sequence[tuple[SeriesWindow, Sequence[dict[str, Any]]]],
    vote_threshold: float = 0.5,
    vote_aggregation: str = "sum",
    merge_gap: int = 0,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Aggregate overlapping slice decisions as point-wise confidence votes.

    Every window covering a point contributes one vote. A window contributes
    its highest anomaly confidence at that point, or zero when it predicts the
    point as normal. ``sum`` preserves a strong single-window detection in an
    overlap region, while ``mean`` requires agreement from the covering windows.
    With zero overlap the two modes are identical.
    """

    if series_length <= 0:
        raise ValueError("series_length must be positive")
    if not 0.0 < vote_threshold <= 1.0:
        raise ValueError("vote_threshold must be in (0, 1]")
    if vote_aggregation not in {"mean", "sum"}:
        raise ValueError("vote_aggregation must be 'mean' or 'sum'")
    coverage_count = np.zeros(series_length, dtype=np.int32)
    confidence_sum = np.zeros(series_length, dtype=np.float64)
    mapped_predictions: list[dict[str, Any]] = []

    for window, raw_predictions in windows_and_predictions:
        start, end = window.interval.start, window.interval.end
        coverage_count[start : end + 1] += 1
        local_scores = np.zeros(window.length, dtype=np.float64)
        mapped = map_window_predictions(window, raw_predictions)
        mapped_predictions.extend(mapped)
        for item in mapped:
            try:
                confidence = float(item.get("confidence", 0.0))
            except (TypeError, ValueError):
                confidence = 0.0
            if not np.isfinite(confidence):
                confidence = 0.0
            confidence = min(1.0, max(0.0, confidence))
            local_start = int(item["start"]) - start
            local_end = int(item["end"]) - start
            local_scores[local_start : local_end + 1] = np.maximum(
                local_scores[local_start : local_end + 1],
                confidence,
            )
        confidence_sum[start : end + 1] += local_scores

    mean_scores = np.divide(
        confidence_sum,
        coverage_count,
        out=np.zeros_like(confidence_sum),
        where=coverage_count > 0,
    )
    point_scores = confidence_sum if vote_aggregation == "sum" else mean_scores
    voted = point_scores >= float(vote_threshold)
    voted_intervals = merge_intervals(labels_to_intervals(voted), gap=max(0, int(merge_gap)))
    predictions: list[dict[str, Any]] = []
    for interval in voted_intervals:
        supporters = [
            item
            for item in mapped_predictions
            if int(item["start"]) <= interval.end and interval.start <= int(item["end"])
        ]
        representative = max(
            supporters,
            key=lambda item: float(item.get("confidence", 0.0)),
        )
        scores = point_scores[interval.start : interval.end + 1]
        normalized_scores = mean_scores[interval.start : interval.end + 1]
        supporting_windows = sorted({int(item["window_index"]) for item in supporters})
        evidence = str(representative.get("evidence", ""))
        item = {
            "start": interval.start,
            "end": interval.end,
            "confidence": round(min(1.0, float(np.max(scores))), 6),
            "vote_score": round(float(np.max(scores)), 6),
            "mean_vote_confidence": round(float(np.mean(normalized_scores)), 6),
            "anomaly_type": str(representative.get("anomaly_type", "unknown")),
            "evidence": (
                f"Global window vote from slices {supporting_windows}; "
                f"representative local evidence: {evidence}"
            )[:1600],
            "source": "window-confidence-vote",
            "supporting_windows": supporting_windows,
            "supporting_prediction_count": len(supporters),
        }
        if isinstance(representative.get("boundary_refinement"), dict):
            item["boundary_refinement"] = representative["boundary_refinement"]
        predictions.append(item)

    diagnostics = {
        "mapped_prediction_count": len(mapped_predictions),
        "voted_prediction_count": len(predictions),
        "voted_anomaly_points": int(voted.sum()),
        "vote_threshold": float(vote_threshold),
        "vote_aggregation": vote_aggregation,
        "maximum_vote_score": float(point_scores.max(initial=0.0)),
        "covered_points": int((coverage_count > 0).sum()),
        "maximum_window_votes_per_point": int(coverage_count.max(initial=0)),
    }
    return predictions, diagnostics
