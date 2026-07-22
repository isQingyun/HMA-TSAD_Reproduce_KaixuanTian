from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

from .intervals import Interval


SAFE_METADATA_KEYS = {
    "dataset_id",
    "description",
    "dimensions",
    "frequency",
    "is_train",
    "length",
    "means",
    "periods",
    "sampling_rate",
    "stationarities",
    "stddevs",
    "trends",
}


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


def _safe_sample_metadata(raw: Any, sample_id: str) -> dict[str, Any]:
    """Select one sample's non-label metadata and discard label-derived fields."""

    if not isinstance(raw, dict):
        return {}
    selected: Any = None
    for key, value in raw.items():
        if str(key) == str(sample_id):
            selected = value
            break
    if selected is None:
        candidates = [value for key, value in raw.items() if str(key) != "mapping"]
        if len(candidates) == 1:
            selected = candidates[0]
    if isinstance(selected, list):
        mappings = [item for item in selected if isinstance(item, dict)]
        selected = next((item for item in mappings if item.get("is_train") is False), mappings[0] if mappings else {})
    if not isinstance(selected, dict):
        selected = raw if any(key in SAFE_METADATA_KEYS for key in raw) else {}
    return {str(key): value for key, value in selected.items() if str(key) in SAFE_METADATA_KEYS}


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
    raw_metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata = _safe_sample_metadata(raw_metadata, sample_id)
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


def metadata_period(metadata: dict[str, Any]) -> int | None:
    raw = metadata.get("periods")
    if isinstance(raw, dict):
        raw = raw.get("value", raw.get("count"))
    try:
        period = int(round(float(raw)))
    except (TypeError, ValueError):
        return None
    return period if period > 1 else None


def estimate_period(values: np.ndarray, metadata: dict[str, Any] | None = None) -> int | None:
    """Estimate a dominant period without labels, preferring safe dataset metadata."""

    known = metadata_period(metadata or {})
    if known is not None and known < len(values) // 2:
        return known
    series = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(series) < 16:
        return None
    series = series[: min(len(series), 8192)]
    x = np.arange(len(series), dtype=np.float64)
    slope, intercept = np.polyfit(x, series, 1)
    centered = series - (slope * x + intercept)
    energy = float(np.dot(centered, centered))
    if not np.isfinite(energy) or energy < 1e-12:
        return None
    fft_size = 1 << (2 * len(centered) - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=fft_size)
    autocorrelation = np.fft.irfft(spectrum * np.conjugate(spectrum), n=fft_size)[: len(centered)]
    autocorrelation /= autocorrelation[0]
    max_lag = min(2048, len(centered) // 3)
    if max_lag < 3:
        return None
    peaks = [
        lag
        for lag in range(2, max_lag)
        if autocorrelation[lag] >= autocorrelation[lag - 1]
        and autocorrelation[lag] > autocorrelation[lag + 1]
        and autocorrelation[lag] > 0.1
    ]
    if not peaks:
        return None
    best_value = max(float(autocorrelation[lag]) for lag in peaks)
    plausible = [lag for lag in peaks if float(autocorrelation[lag]) >= 0.9 * best_value]
    return int(min(plausible)) if plausible else None


def analysis_window_points(
    sample: TimeSeriesSample,
    min_points: int = 256,
    max_points: int = 2048,
    cycles: float = 3.0,
) -> tuple[int, int | None]:
    source = sample.train_values if sample.train_values is not None else sample.values
    period = estimate_period(source, sample.metadata)
    if period is not None:
        desired = int(round(max(1.0, cycles) * period))
    else:
        desired = int(round(len(source) * 0.20))
    window = max(1, min(len(source), max(min_points, min(max_points, desired))))
    return window, period


def multiscale_window_points(
    sample: TimeSeriesSample,
    cycles: Sequence[float],
    min_points: int = 256,
    max_points: int = 2048,
    fallback_fraction: float = 0.20,
    fallback_cycles: float = 3.0,
) -> tuple[list[int], int | None]:
    """Map cycle counts to window lengths while preserving their relative scales.

    When a period is available, every window is the requested number of periods
    (limited only by the source length).  When period estimation fails, the old
    20%-of-series analysis window is treated as a three-cycle fallback and the
    remaining scales are derived from it.  This prevents 1P/3P/5P from silently
    collapsing to one length because of a shared minimum-point clamp.
    """

    requested = [float(value) for value in cycles]
    if not requested or any(not np.isfinite(value) or value <= 0.0 for value in requested):
        raise ValueError("cycles must contain one or more positive finite values")
    source = sample.train_values if sample.train_values is not None else sample.values
    period = estimate_period(source, sample.metadata)
    if period is not None:
        base_period = float(period)
    else:
        fallback = int(round(len(source) * fallback_fraction))
        fallback = max(1, min(len(source), max(min_points, min(max_points, fallback))))
        base_period = fallback / max(1.0, float(fallback_cycles))
    windows = [max(1, min(len(source), int(round(value * base_period)))) for value in requested]
    return windows, period


def _segment_features(segment: np.ndarray) -> np.ndarray:
    values = np.asarray(segment, dtype=np.float64).reshape(-1)
    center = float(np.median(values))
    mad = float(np.median(np.abs(values - center)))
    amplitude = float(np.quantile(values, 0.95) - np.quantile(values, 0.05))
    roughness = float(np.median(np.abs(np.diff(values)))) if len(values) > 1 else 0.0
    curvature = float(np.median(np.abs(np.diff(values, n=2)))) if len(values) > 2 else 0.0
    slope = float((values[-1] - values[0]) / max(1, len(values) - 1))
    # Quantiles and median derivatives intentionally ignore isolated impulses.
    # Explicit extremes keep such events visible in reference profiles.
    lower_extreme = float(center - np.min(values))
    upper_extreme = float(np.max(values) - center)
    return np.asarray(
        [center, mad, amplitude, roughness, curvature, slope, lower_extreme, upper_extreme],
        dtype=np.float64,
    )


def reference_catalog(
    series: np.ndarray,
    intervals: list[Interval],
    cycle_scales: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    if cycle_scales is not None and len(cycle_scales) != len(intervals):
        raise ValueError("cycle_scales must align one-to-one with intervals")
    catalog: list[dict[str, Any]] = []
    for index, interval in enumerate(intervals):
        segment = np.asarray(series[interval.start : interval.end + 1], dtype=np.float64)
        features = _segment_features(segment)
        item = {
                "reference_index": index,
                "interval": interval.to_list(),
                "length": interval.length,
                "median": round(float(features[0]), 6),
                "mad": round(float(features[1]), 6),
                "amplitude_90": round(float(features[2]), 6),
                "roughness": round(float(features[3]), 6),
                "curvature": round(float(features[4]), 6),
                "slope_per_point": round(float(features[5]), 8),
                "lower_extreme": round(float(features[6]), 6),
                "upper_extreme": round(float(features[7]), 6),
            }
        if cycle_scales is not None:
            item["window_cycles"] = float(cycle_scales[index])
        catalog.append(item)
    return catalog


def shape_distance(first: np.ndarray, second: np.ndarray, points: int = 128) -> float:
    """Return a label-free, scale-robust shape distance; lower is more similar."""

    def vector(values: np.ndarray) -> np.ndarray:
        normalized, _ = robust_standardize(np.asarray(values, dtype=np.float64).reshape(-1))
        if len(normalized) == 1:
            return np.repeat(normalized, points)
        old_x = np.linspace(0.0, 1.0, num=len(normalized))
        new_x = np.linspace(0.0, 1.0, num=points)
        return np.interp(new_x, old_x, normalized)

    a = vector(first)
    b = vector(second)
    level = float(np.mean(np.abs(a - b)))
    derivative = float(np.mean(np.abs(np.diff(a) - np.diff(b))))
    return level + 0.35 * derivative


def select_reference_intervals(
    sample: TimeSeriesSample,
    count: int = 3,
    window_fraction: float = 0.20,
    window_points: int | Sequence[int] | None = None,
) -> tuple[np.ndarray, list[Interval], str]:
    """Select diverse consensus references, including mixed window lengths."""

    source = sample.train_values if sample.train_values is not None else sample.values
    source_name = "train" if sample.train_values is not None else "test-unsupervised"
    normalized, _ = robust_standardize(source)
    length = len(normalized)
    if window_points is None:
        default_window = max(256, min(length, 2048, int(round(length * window_fraction))))
        windows = [default_window] * max(1, int(count))
    elif isinstance(window_points, (int, np.integer)):
        windows = [max(1, min(length, int(window_points)))] * max(1, int(count))
    else:
        windows = [max(1, min(length, int(value))) for value in window_points]
        if not windows:
            raise ValueError("window_points cannot be empty")

    # Training series are explicitly the anomaly-free context in these benchmarks.
    # Use broad, evenly separated excerpts instead of cherry-picking by a test label
    # or by a low-variance heuristic that can over-represent flat phases.
    if sample.train_values is not None:
        fractions = np.linspace(0.0, 1.0, num=max(2, len(windows)))[: len(windows)]
        references = []
        for fraction, window in zip(fractions, windows):
            start = int(round(float(fraction) * max(0, length - window)))
            references.append(Interval(start, start + window - 1))
        return normalized, references, source_name

    # For test-only datasets, find consensus candidates separately at each
    # requested scale. Duplicate scales (the two 3P references) are kept apart.
    selected: list[Interval] = []
    selected_by_window: dict[int, list[Interval]] = {}
    score_cache: dict[int, list[tuple[float, int]]] = {}
    for window in windows:
        if window >= length:
            selected.append(Interval(0, length - 1))
            continue
        if window not in score_cache:
            candidate_starts = np.linspace(
                0,
                length - window,
                num=min(48, max(5, length // window + 1)),
                dtype=int,
            )
            starts = sorted(set(int(value) for value in candidate_starts))
            features = np.vstack([_segment_features(normalized[start : start + window]) for start in starts])
            center = np.median(features, axis=0)
            scale = np.median(np.abs(features - center), axis=0)
            scale = np.where(scale < 1e-9, 1.0, scale)
            score_cache[window] = sorted(
                (float(np.mean(np.abs((features[index] - center) / scale))), start)
                for index, start in enumerate(starts)
            )
        same_scale = selected_by_window.setdefault(window, [])
        chosen: Interval | None = None
        for _, start in score_cache[window]:
            candidate = Interval(start, start + window - 1)
            if all(candidate.intersection_length(existing) < max(1, window // 4) for existing in same_scale):
                chosen = candidate
                break
        if chosen is None:
            chosen = Interval(score_cache[window][0][1], score_cache[window][0][1] + window - 1)
        same_scale.append(chosen)
        selected.append(chosen)
    return normalized, selected, source_name
