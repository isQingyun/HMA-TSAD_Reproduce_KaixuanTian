from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class Interval:
    """Inclusive integer interval."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0:
            raise ValueError("Interval start must be non-negative")
        if self.end < self.start:
            raise ValueError("Interval end must be >= start")

    @property
    def length(self) -> int:
        return self.end - self.start + 1

    def overlaps(self, other: "Interval") -> bool:
        return self.start <= other.end and other.start <= self.end

    def intersection_length(self, other: "Interval") -> int:
        return max(0, min(self.end, other.end) - max(self.start, other.start) + 1)

    def clamp(self, length: int) -> "Interval":
        if length <= 0:
            raise ValueError("Series length must be positive")
        start = min(max(0, self.start), length - 1)
        end = min(max(start, self.end), length - 1)
        return Interval(start, end)

    def to_list(self) -> list[int]:
        return [self.start, self.end]


def merge_intervals(intervals: Iterable[Interval], gap: int = 0) -> list[Interval]:
    ordered = sorted(intervals)
    if not ordered:
        return []
    merged = [ordered[0]]
    for current in ordered[1:]:
        previous = merged[-1]
        if current.start <= previous.end + gap + 1:
            merged[-1] = Interval(previous.start, max(previous.end, current.end))
        else:
            merged.append(current)
    return merged


def coverage_fraction(intervals: Iterable[Interval], length: int) -> float:
    if length <= 0:
        return 0.0
    covered = sum(item.length for item in merge_intervals(intervals))
    return min(1.0, covered / length)


def labels_to_intervals(labels: Sequence[int] | np.ndarray) -> list[Interval]:
    binary = np.asarray(labels).reshape(-1) > 0
    result: list[Interval] = []
    start: int | None = None
    for index, value in enumerate(binary):
        if value and start is None:
            start = index
        if start is not None and (not value or index == len(binary) - 1):
            end = index if value and index == len(binary) - 1 else index - 1
            result.append(Interval(start, end))
            start = None
    return result


def intervals_to_mask(intervals: Iterable[Interval], length: int) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for interval in intervals:
        item = interval.clamp(length)
        mask[item.start : item.end + 1] = True
    return mask

