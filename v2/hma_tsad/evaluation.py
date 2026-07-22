from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .intervals import Interval, intervals_to_mask, labels_to_intervals
from .state import save_json


def _safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return _safe_ratio(2.0 * precision * recall, precision + recall)


def _prediction_intervals(payload: dict[str, Any], length: int) -> list[Interval]:
    result: list[Interval] = []
    for item in payload.get("predictions", []):
        try:
            start = int(item["start"])
            end = int(item["end"])
            if end < start:
                start, end = end, start
            result.append(Interval(max(0, start), max(0, end)).clamp(length))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def compute_metrics(labels: np.ndarray, predictions: list[Interval]) -> dict[str, float | int]:
    truth_mask = np.asarray(labels).reshape(-1) > 0
    length = len(truth_mask)
    prediction_mask = intervals_to_mask(predictions, length)
    true_intervals = labels_to_intervals(truth_mask)

    tp = int(np.logical_and(truth_mask, prediction_mask).sum())
    fp = int(np.logical_and(~truth_mask, prediction_mask).sum())
    fn = int(np.logical_and(truth_mask, ~prediction_mask).sum())
    point_precision = _safe_ratio(tp, tp + fp)
    point_recall = _safe_ratio(tp, tp + fn)

    adjusted = prediction_mask.copy()
    for truth in true_intervals:
        if prediction_mask[truth.start : truth.end + 1].any():
            adjusted[truth.start : truth.end + 1] = True
    adjusted_tp = int(np.logical_and(truth_mask, adjusted).sum())
    adjusted_fp = int(np.logical_and(~truth_mask, adjusted).sum())
    adjusted_fn = int(np.logical_and(truth_mask, ~adjusted).sum())
    pa_precision = _safe_ratio(adjusted_tp, adjusted_tp + adjusted_fp)
    pa_recall = _safe_ratio(adjusted_tp, adjusted_tp + adjusted_fn)

    range_recall = _safe_ratio(
        sum(any(truth.overlaps(prediction) for prediction in predictions) for truth in true_intervals),
        len(true_intervals),
    )
    range_precision = _safe_ratio(
        sum(any(prediction.overlaps(truth) for truth in true_intervals) for prediction in predictions),
        len(predictions),
    )
    overlap_recall = _safe_ratio(
        sum(
            min(1.0, sum(truth.intersection_length(prediction) for prediction in predictions) / truth.length)
            for truth in true_intervals
        ),
        len(true_intervals),
    )
    overlap_precision = _safe_ratio(
        sum(
            min(1.0, sum(prediction.intersection_length(truth) for truth in true_intervals) / prediction.length)
            for prediction in predictions
        ),
        len(predictions),
    )

    return {
        "series_length": length,
        "true_anomaly_points": int(truth_mask.sum()),
        "true_ranges": len(true_intervals),
        "predicted_ranges": len(predictions),
        "point_precision": point_precision,
        "point_recall": point_recall,
        "point_f1": _f1(point_precision, point_recall),
        "point_adjusted_precision": pa_precision,
        "point_adjusted_recall": pa_recall,
        "point_adjusted_f1": _f1(pa_precision, pa_recall),
        "range_precision": range_precision,
        "range_recall": range_recall,
        "range_f1": _f1(range_precision, range_recall),
        "overlap_precision": overlap_precision,
        "overlap_recall": overlap_recall,
        "overlap_f1": _f1(overlap_precision, overlap_recall),
    }


def evaluate_prediction_file(prediction_path: str | Path, data_root: str | Path) -> dict[str, Any]:
    """Load labels only after inference has produced an immutable prediction file."""

    path = Path(prediction_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    dataset = str(payload["dataset"])
    sample_id = str(payload["sample_id"])
    label_path = Path(data_root) / dataset / f"{sample_id}_labels.npy"
    labels = np.load(label_path)
    intervals = _prediction_intervals(payload, len(np.asarray(labels).reshape(-1)))
    metrics = compute_metrics(labels, intervals)
    result: dict[str, Any] = {
        "dataset": dataset,
        "sample_id": sample_id,
        "mode": payload.get("mode"),
        "prediction_path": str(path),
        "label_path": str(label_path),
        "label_usage": "evaluation only; labels were not provided to the agent",
        **metrics,
    }
    save_json(path.parent / "metrics.json", result)
    return result


def write_summary(records: list[dict[str, Any]], output_dir: str | Path) -> dict[str, Any]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    numeric_metrics = [
        "point_precision",
        "point_recall",
        "point_f1",
        "point_adjusted_precision",
        "point_adjusted_recall",
        "point_adjusted_f1",
        "range_precision",
        "range_recall",
        "range_f1",
        "overlap_precision",
        "overlap_recall",
        "overlap_f1",
    ]
    aggregate = {
        metric: float(np.mean([float(item[metric]) for item in records])) if records else 0.0
        for metric in numeric_metrics
    }
    summary = {"sample_count": len(records), "macro_average": aggregate, "records": records}
    save_json(output / "summary.json", summary)
    if records:
        with (output / "summary.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
    return summary

