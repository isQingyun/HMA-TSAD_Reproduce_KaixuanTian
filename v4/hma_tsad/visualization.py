from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .intervals import Interval


def _minmax_downsample(x: np.ndarray, y: np.ndarray, max_points: int) -> tuple[np.ndarray, np.ndarray]:
    if len(y) <= max_points:
        return x, y
    bins = max(2, max_points // 2)
    boundaries = np.linspace(0, len(y), bins + 1, dtype=int)
    selected: list[int] = []
    for left, right in zip(boundaries[:-1], boundaries[1:]):
        if right <= left:
            continue
        segment = y[left:right]
        low = left + int(np.argmin(segment))
        high = left + int(np.argmax(segment))
        selected.extend(sorted({low, high}))
    indices = np.asarray(sorted(set(selected)), dtype=int)
    return x[indices], y[indices]


def render_interval(
    series: np.ndarray,
    interval: Interval,
    output_path: str | Path,
    title: str,
    max_points: int = 1600,
    highlight: Interval | None = None,
) -> Path:
    interval = interval.clamp(len(series))
    x_full = np.arange(interval.start, interval.end + 1)
    y_full = np.asarray(series[interval.start : interval.end + 1])
    x, y = _minmax_downsample(x_full, y_full, max_points)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, axis = plt.subplots(figsize=(18, 6.5), dpi=100)
    axis.plot(x, y, color="#155fa0", linewidth=1.35)
    if highlight is not None:
        highlighted = highlight.clamp(len(series))
        axis.axvspan(
            highlighted.start,
            highlighted.end,
            color="#d62728",
            alpha=0.18,
            zorder=0,
        )
        axis.axvline(highlighted.start, color="#d62728", alpha=0.75, linewidth=1.0)
        axis.axvline(highlighted.end, color="#d62728", alpha=0.75, linewidth=1.0)
        axis.text(
            0.99,
            0.98,
            f"Highlighted candidate: [{highlighted.start}, {highlighted.end}]",
            transform=axis.transAxes,
            va="top",
            ha="right",
            fontsize=10,
            color="#9b1c1c",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#d62728"},
        )
    axis.set_title(title, fontsize=15, pad=12)
    axis.set_xlabel("Global time index (inclusive)")
    axis.set_ylabel("Robust standardized value")
    axis.set_xlim(interval.start, interval.end)
    ticks = np.linspace(interval.start, interval.end, num=min(20, interval.length), dtype=int)
    axis.set_xticks(sorted(set(int(v) for v in ticks)))
    axis.grid(True, color="#c7cdd4", alpha=0.75, linewidth=0.8)
    axis.text(
        0.01,
        0.98,
        f"Exact visible range: [{interval.start}, {interval.end}] | points: {interval.length}",
        transform=axis.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#888888"},
    )
    fig.tight_layout()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    return output


def render_compare(
    series_a: np.ndarray,
    interval_a: Interval,
    series_b: np.ndarray,
    interval_b: Interval,
    output_path: str | Path,
    title_a: str,
    title_b: str,
    max_points: int = 1400,
) -> Path:
    interval_a = interval_a.clamp(len(series_a))
    interval_b = interval_b.clamp(len(series_b))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(18, 8), dpi=100)
    for axis, series, interval, title, color in (
        (axes[0], series_a, interval_a, title_a, "#155fa0"),
        (axes[1], series_b, interval_b, title_b, "#2a8a4a"),
    ):
        x_full = np.arange(interval.start, interval.end + 1)
        y_full = np.asarray(series[interval.start : interval.end + 1])
        x, y = _minmax_downsample(x_full, y_full, max_points)
        axis.plot(x, y, color=color, linewidth=1.25)
        axis.set_title(title, fontsize=12)
        axis.set_xlim(interval.start, interval.end)
        axis.set_ylabel("Robust z")
        ticks = np.linspace(interval.start, interval.end, num=min(9, interval.length), dtype=int)
        axis.set_xticks(sorted(set(int(v) for v in ticks)))
        axis.grid(True, color="#c7cdd4", alpha=0.75, linewidth=0.8)
        axis.text(
            0.01,
            0.96,
            f"Exact range: [{interval.start}, {interval.end}]",
            transform=axis.transAxes,
            va="top",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#888888"},
        )
    axes[-1].set_xlabel("Global/source time index (inclusive)")
    fig.tight_layout()
    fig.savefig(output, format="png", bbox_inches="tight")
    plt.close(fig)
    return output
