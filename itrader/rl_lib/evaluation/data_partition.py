from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


_SPLITS = {"all", "train", "validation", "test"}


@dataclass(frozen=True)
class StartRange:
    symbol: str
    first: int
    last: int

    @property
    def size(self) -> int:
        return max(0, self.last - self.first + 1)


class DataPartition:
    """Own symbol-safe chronological eligibility for episode starts."""

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        window_size: int,
        split: str = "all",
        split_fractions: Iterable[float] = (0.7, 0.15, 0.15),
        purge_rows: int = 0,
        symbol_col: str | None = None,
        timestamp_col: str | None = None,
    ):
        split = str(split).lower()
        if split not in _SPLITS:
            raise ValueError(f"data_split must be one of {sorted(_SPLITS)}, got {split!r}")
        fractions = tuple(float(value) for value in split_fractions)
        if len(fractions) != 3 or any(value <= 0.0 for value in fractions):
            raise ValueError("split_fractions must contain three positive values")
        if not np.isclose(sum(fractions), 1.0):
            raise ValueError("split_fractions must sum to 1.0")

        self.window_size = int(window_size)
        self.split = split
        self.fractions = fractions
        self.purge_rows = max(0, int(purge_rows))
        self._segments = self._contiguous_segments(frame, symbol_col, timestamp_col)

    @staticmethod
    def _contiguous_segments(
        frame: pd.DataFrame,
        symbol_col: str | None,
        timestamp_col: str | None,
    ) -> list[tuple[str, int, int]]:
        if frame.empty:
            raise ValueError("Market dataframe is empty")
        symbols = (
            frame[symbol_col].astype(str).to_numpy(copy=False)
            if symbol_col is not None
            else np.full(len(frame), "__single__", dtype=object)
        )
        timestamps = (
            pd.to_datetime(frame[timestamp_col]).to_numpy(copy=False)
            if timestamp_col is not None
            else None
        )
        starts = [0]
        for index in range(1, len(frame)):
            symbol_changed = symbols[index] != symbols[index - 1]
            time_restarted = timestamps is not None and timestamps[index] <= timestamps[index - 1]
            if symbol_changed or time_restarted:
                starts.append(index)
        starts.append(len(frame))
        return [
            (str(symbols[start]), start, end)
            for start, end in zip(starts[:-1], starts[1:])
            if end > start
        ]

    def _outcome_interval(self, start: int, end: int) -> tuple[int, int]:
        if self.split == "all":
            return start, end
        length = end - start
        train_boundary = start + int(length * self.fractions[0])
        validation_boundary = start + int(length * (self.fractions[0] + self.fractions[1]))
        purge = self.purge_rows
        if self.split == "train":
            return start, max(start, train_boundary - purge)
        if self.split == "validation":
            return min(end, train_boundary + purge), max(start, validation_boundary - purge)
        return min(end, validation_boundary + purge), end

    def valid_ranges(self, horizon: int) -> list[StartRange]:
        horizon = int(horizon)
        if horizon <= 0:
            raise ValueError("Episode horizon must be positive")
        ranges: list[StartRange] = []
        for symbol, segment_start, segment_end in self._segments:
            outcome_start, outcome_end = self._outcome_interval(segment_start, segment_end)
            first = max(segment_start + self.window_size, outcome_start)
            last = outcome_end - horizon
            if last >= first:
                ranges.append(StartRange(symbol=symbol, first=first, last=last))
        if not ranges:
            raise ValueError(
                f"No {self.split!r} episode fits window={self.window_size}, "
                f"horizon={horizon}, purge_rows={self.purge_rows}"
            )
        return ranges

    def sample(self, horizon: int, rng: Any) -> int:
        ranges = self.valid_ranges(horizon)
        total = sum(item.size for item in ranges)
        offset = int(rng.integers(0, total))
        for item in ranges:
            if offset < item.size:
                return item.first + offset
            offset -= item.size
        raise RuntimeError("Failed to sample an eligible episode start")

    def contains(self, start_idx: int, horizon: int) -> bool:
        start_idx = int(start_idx)
        return any(item.first <= start_idx <= item.last for item in self.valid_ranges(horizon))

    def clamp(self, start_idx: int, horizon: int) -> int:
        start_idx = int(start_idx)
        ranges = self.valid_ranges(horizon)
        candidates = [min(max(start_idx, item.first), item.last) for item in ranges]
        return min(candidates, key=lambda value: (abs(value - start_idx), value))

    def symbol_for(self, start_idx: int, horizon: int) -> str:
        start_idx = int(start_idx)
        for item in self.valid_ranges(horizon):
            if item.first <= start_idx <= item.last:
                return item.symbol
        raise ValueError(f"Start index {start_idx} is not eligible for horizon {horizon}")
