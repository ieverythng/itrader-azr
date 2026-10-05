from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator
from zoneinfo import ZoneInfo

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .contracts import (
    HistoricalBarsRequest,
    HistoricalMarketDataSource,
    MarketDataPage,
    SnapshotBuildConfig,
)


_BAR_COLUMNS = {
    "t": "ts",
    "o": "OPEN",
    "h": "HIGH",
    "l": "LOW",
    "c": "CLOSE",
    "v": "VOLUME",
    "n": "trade_count",
    "vw": "vwap",
}
_SESSIONS = ("pre_market", "regular", "after_hours")


@dataclass(frozen=True)
class SnapshotBuildResult:
    snapshot_root: Path
    bars_path: Path
    manifest_path: Path


def _year_ranges(
    start: date,
    end: date,
    timezone_name: str,
) -> Iterator[tuple[datetime, datetime]]:
    local_timezone = ZoneInfo(timezone_name)
    cursor = start
    while cursor <= end:
        next_year = date(cursor.year + 1, 1, 1)
        stop = min(next_year, end + timedelta(days=1))
        request_end = (
            datetime.combine(end, time(20, 0), local_timezone)
            if stop == end + timedelta(days=1)
            else datetime.combine(stop, time.min, local_timezone)
        )
        yield (
            datetime.combine(cursor, time.min, local_timezone).astimezone(timezone.utc),
            request_end.astimezone(timezone.utc),
        )
        cursor = stop


def _parse_clock(value: str) -> int:
    parsed = datetime.strptime(value, "%H:%M")
    return parsed.hour * 60 + parsed.minute


def _calendar_frame(payload: Any) -> pd.DataFrame:
    if not isinstance(payload, list) or not payload:
        raise ValueError("Market calendar response is empty or malformed")
    calendar = pd.DataFrame(payload)
    required = {"date", "open", "close"}
    if not required <= set(calendar.columns):
        raise ValueError(f"Market calendar is missing fields: {sorted(required - set(calendar.columns))}")
    calendar = calendar.loc[:, ["date", "open", "close"]].copy()
    calendar["trading_date"] = calendar["date"].astype(str)
    calendar["open_minute"] = calendar["open"].map(_parse_clock)
    calendar["close_minute"] = calendar["close"].map(_parse_clock)
    if calendar["trading_date"].duplicated().any():
        raise ValueError("Market calendar contains duplicate trading dates")
    return calendar.loc[:, ["trading_date", "open_minute", "close_minute"]]


def _normalize_bars(
    pages: list[MarketDataPage],
    symbol: str,
    calendar: pd.DataFrame,
    timezone_name: str,
) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for page in pages:
        payload = page.payload
        if not isinstance(payload, dict) or not isinstance(payload.get("bars"), dict):
            raise ValueError(f"Malformed bars response for {symbol}")
        records.extend(payload["bars"].get(symbol, []))
    if not records:
        return pd.DataFrame()

    frame = pd.DataFrame.from_records(records).rename(columns=_BAR_COLUMNS)
    required = set(_BAR_COLUMNS.values())
    if not required <= set(frame.columns):
        raise ValueError(f"Bars for {symbol} are missing fields: {sorted(required - set(frame.columns))}")
    frame = frame.loc[:, list(_BAR_COLUMNS.values())].copy()
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True, errors="raise")
    frame["symbol"] = symbol
    if frame.duplicated(["symbol", "ts"]).any():
        raise ValueError(f"Duplicate one-minute bars returned for {symbol}")

    local = frame["ts"].dt.tz_convert(ZoneInfo(timezone_name))
    frame["trading_date"] = local.dt.date.astype(str)
    frame["local_minute"] = local.dt.hour * 60 + local.dt.minute
    frame = frame.merge(calendar, how="left", on="trading_date", validate="many_to_one")
    if frame["open_minute"].isna().any():
        missing = sorted(frame.loc[frame["open_minute"].isna(), "trading_date"].unique())
        raise ValueError(f"Bars fall outside the supplied exchange calendar: {missing[:5]}")

    minute = frame["local_minute"].to_numpy()
    market_open = frame["open_minute"].to_numpy()
    market_close = frame["close_minute"].to_numpy()
    frame["session"] = np.select(
        [
            (minute >= 4 * 60) & (minute < market_open),
            (minute >= market_open) & (minute < market_close),
            (minute >= market_close) & (minute < 20 * 60),
        ],
        list(_SESSIONS),
        default="outside",
    )
    frame = frame.loc[frame["session"] != "outside"].copy()
    frame["session_id"] = (
        frame["symbol"] + ":" + frame["trading_date"] + ":" + frame["session"]
    )
    frame["year"] = frame["ts"].dt.year.astype("int16")
    frame = frame.drop(columns=["local_minute", "open_minute", "close_minute"])
    return frame.sort_values(["symbol", "ts"], kind="stable").reset_index(drop=True)


def _write_raw(path: Path, page: MarketDataPage) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", encoding="utf-8") as stream:
        json.dump(
            {
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "request": page.request,
                "payload": page.payload,
            },
            stream,
            sort_keys=True,
            separators=(",", ":"),
        )
    temporary.replace(path)


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _read_raw(path: Path) -> MarketDataPage:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        record = json.load(stream)
    return MarketDataPage(request=record["request"], payload=record["payload"])


def _write_partitioned(frame: pd.DataFrame, bars_root: Path, part_index: int) -> None:
    if frame.empty:
        return
    for keys, partition in frame.groupby(["session", "symbol", "year"], sort=True):
        session, symbol, year = keys
        folder = bars_root / f"session={session}" / f"symbol={symbol}" / f"year={int(year)}"
        folder.mkdir(parents=True, exist_ok=True)
        stored = partition.drop(columns=["session", "symbol", "year"])
        table = pa.Table.from_pandas(stored, preserve_index=False)
        pq.write_table(
            table,
            folder / f"part-{part_index:05d}.parquet",
            compression="zstd",
            write_statistics=True,
        )


def _completion_marker(
    root: Path,
    *,
    part_index: int,
    symbol: str,
    year: int,
) -> Path:
    return root / "completed" / f"{part_index:05d}_{symbol}_{year}.json"


def _clear_incomplete_unit(
    bars_root: Path,
    raw_root: Path,
    *,
    part_index: int,
    symbol: str,
    year: int,
) -> None:
    for path in bars_root.rglob(f"part-{part_index:05d}.parquet"):
        path.unlink()
    for path in raw_root.glob(f"*_{symbol}_{year}_bars.json.gz"):
        path.unlink()


def _write_completion_marker(
    path: Path,
    *,
    part_index: int,
    symbol: str,
    year: int,
    row_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "part_index": part_index,
                "symbol": symbol,
                "year": year,
                "row_count": row_count,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _content_hash(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        if path.name == "manifest.json" or path.name.endswith(".tmp"):
            continue
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_market_snapshot(
    source: HistoricalMarketDataSource,
    config: SnapshotBuildConfig,
    *,
    progress: Any = None,
) -> SnapshotBuildResult:
    """Build one immutable raw-plus-normalized historical snapshot."""
    root = config.snapshot_root
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Snapshot root already exists; snapshots are immutable: {root}")
    root.mkdir(parents=True, exist_ok=True)
    build_request = {
        "schema_version": config.schema_version,
        "provider": source.provider,
        "symbols": list(config.symbols),
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "feed": config.feed,
        "adjustment": config.adjustment,
        "timeframe": config.timeframe,
        "timezone": config.timezone_name,
        "usage_classification": config.usage_classification,
    }
    build_request_path = root / "build_request.json"
    if build_request_path.exists():
        previous_request = json.loads(build_request_path.read_text(encoding="utf-8"))
        if previous_request != build_request:
            raise ValueError("Incomplete snapshot can only resume with its original build request")
    else:
        _write_json_atomic(build_request_path, build_request)
    raw_root = root / "raw" / source.provider
    bars_root = root / "bars"

    calendar_path = raw_root / "000000_calendar.json.gz"
    if calendar_path.exists():
        calendar_page = _read_raw(calendar_path)
    else:
        calendar_page = source.fetch_calendar(config.start, config.end)
        _write_raw(calendar_path, calendar_page)
    calendar = _calendar_frame(calendar_page.payload)
    existing_raw = sorted(raw_root.glob("*.json.gz"))
    raw_index = max(
        (int(path.name.split("_", 1)[0]) for path in existing_raw),
        default=0,
    ) + 1
    part_index = 0
    notify = progress or (lambda _message: None)
    notify(f"calendar {config.start}..{config.end}: {len(calendar)} sessions")

    for symbol in config.symbols:
        for request_start, request_end in _year_ranges(
            config.start,
            config.end,
            config.timezone_name,
        ):
            marker = _completion_marker(
                root,
                part_index=part_index,
                symbol=symbol,
                year=request_start.year,
            )
            if marker.is_file():
                notify(f"{symbol} {request_start.year}: already complete; resuming")
                part_index += 1
                continue
            _clear_incomplete_unit(
                bars_root,
                raw_root,
                part_index=part_index,
                symbol=symbol,
                year=request_start.year,
            )
            request = HistoricalBarsRequest(
                symbol=symbol,
                start=request_start,
                end=request_end,
                timeframe=config.timeframe,
                feed=config.feed,
                adjustment=config.adjustment,
            )
            pages = list(source.iter_bar_pages(request))
            for page in pages:
                _write_raw(
                    raw_root / f"{raw_index:06d}_{symbol}_{request_start.year}_bars.json.gz",
                    page,
                )
                raw_index += 1
            frame = _normalize_bars(pages, symbol, calendar, config.timezone_name)
            _write_partitioned(frame, bars_root, part_index)
            _write_completion_marker(
                marker,
                part_index=part_index,
                symbol=symbol,
                year=request_start.year,
                row_count=len(frame),
            )
            part_index += 1
            notify(
                f"{symbol} {request_start.year}: {len(frame):,} bars "
                f"({len(pages)} pages)"
            )

    parquet_glob = (bars_root / "**" / "*.parquet").as_posix().replace("'", "''")
    with duckdb.connect(":memory:") as connection:
        counts = connection.execute(
            f"SELECT session, COUNT(*) FROM read_parquet('{parquet_glob}', hive_partitioning=true) "
            "GROUP BY session ORDER BY session"
        ).fetchall()
    session_counts = {str(session): int(count) for session, count in counts}
    row_count = sum(session_counts.values())
    if row_count == 0:
        raise ValueError("Snapshot contains no in-session bars")
    raw_paths = sorted(raw_root.glob("*.json.gz"))
    manifest = {
        "schema_version": config.schema_version,
        "dataset_id": root.name,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "provider": source.provider,
        "feed": config.feed,
        "adjustment": config.adjustment,
        "timeframe": config.timeframe,
        "timezone": config.timezone_name,
        "usage_classification": config.usage_classification,
        "start": config.start.isoformat(),
        "end": config.end.isoformat(),
        "symbols": list(config.symbols),
        "row_count": row_count,
        "session_counts": {key: session_counts.get(key, 0) for key in sorted(_SESSIONS)},
        "raw_page_count": len(list(raw_root.glob("*_bars.json.gz"))),
        "raw_files": [
            {
                "path": path.relative_to(root).as_posix(),
                "sha256": _sha256_file(path),
            }
            for path in raw_paths
        ],
        "partitioning": ["session", "symbol", "year"],
        "missing_bar_policy": "provider_omission_preserved_no_forward_fill",
        "corporate_action_policy": {
            "prices": "split_adjusted",
            "dividends": "not_applied",
            "raw_provider_payloads_retained": True,
        },
        "content_sha256": _content_hash(root),
    }
    _write_json_atomic(manifest_path, manifest)
    notify(f"complete: {row_count:,} bars -> {root}")
    return SnapshotBuildResult(root, bars_root, manifest_path)


def open_snapshot_catalog(snapshot_root: Path | str) -> duckdb.DuckDBPyConnection:
    """Open a rebuildable in-memory DuckDB catalog over immutable Parquet."""
    root = Path(snapshot_root).expanduser().resolve(strict=True)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest}")
    parquet_glob = (root / "bars" / "**" / "*.parquet").as_posix().replace("'", "''")
    connection = duckdb.connect(":memory:")
    connection.execute(
        f"CREATE VIEW bars AS SELECT * FROM read_parquet('{parquet_glob}', hive_partitioning=true)"
    )
    connection.execute("CREATE TABLE snapshot_manifest AS SELECT ? AS payload", [manifest.read_text(encoding="utf-8")])
    return connection
