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
        boundary_cols: Iterable[str] = (),
        expected_bar_interval: str | pd.Timedelta | None = None,
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
        self._symbol_ranges = self._contiguous_symbol_ranges(frame, symbol_col)
        self._segments = self._contiguous_segments(
            frame,
            symbol_col,
            timestamp_col,
            tuple(boundary_cols),
            expected_bar_interval,
        )

    @staticmethod
    def _symbol_codes(
        frame: pd.DataFrame,
        symbol_col: str | None,
    ) -> tuple[np.ndarray, list[str]]:
        if symbol_col is None:
            return np.zeros(len(frame), dtype=np.int8), ["__single__"]
        categorical = pd.Categorical(frame[symbol_col].astype(str))
        return categorical.codes, [str(value) for value in categorical.categories]

    @classmethod
    def _contiguous_symbol_ranges(
        cls,
        frame: pd.DataFrame,
        symbol_col: str | None,
    ) -> dict[str, tuple[int, int]]:
        if frame.empty:
            raise ValueError("Market dataframe is empty")
        symbol_codes, labels = cls._symbol_codes(frame, symbol_col)
        breaks = np.flatnonzero(
            np.r_[True, symbol_codes[1:] != symbol_codes[:-1], True]
        )
        ranges: dict[str, tuple[int, int]] = {}
        for start, end in zip(breaks[:-1], breaks[1:]):
            symbol = labels[int(symbol_codes[start])]
            if symbol in ranges:
                raise ValueError("Market dataframe must group each symbol contiguously")
            ranges[symbol] = (int(start), int(end))
        return ranges

    @staticmethod
    def _contiguous_segments(
        frame: pd.DataFrame,
        symbol_col: str | None,
        timestamp_col: str | None,
        boundary_cols: tuple[str, ...],
        expected_bar_interval: str | pd.Timedelta | None,
    ) -> list[tuple[str, int, int]]:
        if frame.empty:
            raise ValueError("Market dataframe is empty")
        missing_boundaries = [column for column in boundary_cols if column not in frame.columns]
        if missing_boundaries:
            raise ValueError(f"Episode boundary columns not in dataframe: {missing_boundaries}")
        symbol_codes, labels = DataPartition._symbol_codes(frame, symbol_col)
        timestamps = (
            pd.to_datetime(frame[timestamp_col], utc=True).array.asi8
            if timestamp_col is not None
            else None
        )
        boundaries = [pd.Categorical(frame[column]).codes for column in boundary_cols]
        expected_delta = (
            int(pd.Timedelta(expected_bar_interval).value)
            if expected_bar_interval is not None
            else None
        )
        breaks = symbol_codes[1:] != symbol_codes[:-1]
        if timestamps is not None:
            deltas = timestamps[1:] - timestamps[:-1]
            breaks |= deltas <= 0
            if expected_delta is not None:
                breaks |= deltas != expected_delta
        for values in boundaries:
            breaks |= values[1:] != values[:-1]
        starts = np.r_[0, np.flatnonzero(breaks) + 1, len(frame)]
        return [
            (labels[int(symbol_codes[start])], int(start), int(end))
            for start, end in zip(starts[:-1], starts[1:])
            if end > start
        ]

    def _outcome_interval(self, symbol: str) -> tuple[int, int]:
        start, end = self._symbol_ranges[symbol]
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
            split_start, split_end = self._outcome_interval(symbol)
            outcome_start = max(segment_start, split_start)
            outcome_end = min(segment_end, split_end)
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
