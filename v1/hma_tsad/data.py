from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .intervals import Interval


@dataclass(frozen=True)
class TimeSeriesSample:
    dataset: str
    sample_id: str
    values: np.ndarray
    train_values: np.ndarray | None
    metadata: dict[str, Any]

    @property
    def length(self) -> int:
        return int(self.values.shape[0])


def _flatten_univariate(array: np.ndarray, source: Path) -> np.ndarray:
    values = np.asarray(array, dtype=np.float64)
    if values.ndim == 2 and values.shape[1] == 1:
        values = values[:, 0]
    if values.ndim != 1:
        raise ValueError(f"Expected a univariate array in {source}, got {values.shape}")
    if not np.all(np.isfinite(values)):
        finite = np.isfinite(values)
        replacement = float(np.median(values[finite])) if finite.any() else 0.0
        values = values.copy()
        values[~finite] = replacement
    return values


def load_sample(data_root: str | Path, dataset: str, sample_id: str) -> TimeSeriesSample:
    """Load only values and metadata. Ground-truth labels are deliberately excluded."""

    dataset_dir = Path(data_root) / dataset
    test_path = dataset_dir / f"{sample_id}_test.npy"
    if not test_path.exists():
        raise FileNotFoundError(test_path)
    values = _flatten_univariate(np.load(test_path), test_path)
    train_path = dataset_dir / f"{sample_id}_train.npy"
    train_values = _flatten_univariate(np.load(train_path), train_path) if train_path.exists() else None
    metadata_path = dataset_dir / "meta_data.yaml"
    metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    if not isinstance(metadata, dict):
        metadata = {"raw_metadata": metadata}
    return TimeSeriesSample(dataset, sample_id, values, train_values, metadata)


def robust_standardize(values: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    values = np.asarray(values, dtype=np.float64)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    scale = 1.4826 * mad
    if not np.isfinite(scale) or scale < 1e-12:
        scale = float(np.std(values))
    if not np.isfinite(scale) or scale < 1e-12:
        scale = 1.0
    normalized = (values - center) / scale
    return normalized, {"center": center, "scale": scale}


def select_reference_intervals(
    sample: TimeSeriesSample,
    count: int = 2,
    window_fraction: float = 0.20,
) -> tuple[np.ndarray, list[Interval], str]:
    """Select low-variation reference candidates without reading test labels."""

    source = sample.train_values if sample.train_values is not None else sample.values
    source_name = "train" if sample.train_values is not None else "test-unsupervised"
    normalized, _ = robust_standardize(source)
    length = len(normalized)
    window = max(256, min(length, 2048, int(round(length * window_fraction))))
    if window >= length:
        return normalized, [Interval(0, length - 1)], source_name

    # Training series are explicitly the anomaly-free context in these benchmarks.
    # Use broad, evenly separated excerpts instead of cherry-picking by a test label
    # or by a low-variance heuristic that can over-represent flat phases.
    if sample.train_values is not None:
        starts = np.linspace(0, length - window, num=max(2, count), dtype=int)
        references = [Interval(int(start), int(start) + window - 1) for start in starts[:count]]
        return normalized, references, source_name

    candidate_starts = np.linspace(0, length - window, num=min(25, max(3, length // window + 1)), dtype=int)
    scores: list[tuple[float, int]] = []
    global_center = float(np.median(normalized))
    for start in sorted(set(int(v) for v in candidate_starts)):
        segment = normalized[start : start + window]
        level = abs(float(np.median(segment)) - global_center)
        roughness = float(np.median(np.abs(np.diff(segment)))) if len(segment) > 1 else 0.0
        amplitude = float(np.quantile(segment, 0.95) - np.quantile(segment, 0.05))
        scores.append((level + 0.35 * roughness + 0.05 * amplitude, start))

    selected: list[Interval] = []
    for _, start in sorted(scores):
        candidate = Interval(start, start + window - 1)
        if all(candidate.intersection_length(existing) < window // 4 for existing in selected):
            selected.append(candidate)
        if len(selected) >= count:
            break
    return normalized, selected, source_name
