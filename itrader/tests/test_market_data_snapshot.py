from __future__ import annotations

import gzip
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from itrader.inputs.market_data import snapshot as snapshot_module

from itrader.inputs.market_data import (
    AlpacaMarketDataSource,
    HistoricalBarsRequest,
    MarketDataPage,
    SnapshotBuildConfig,
    build_market_snapshot,
    open_snapshot_catalog,
)


class _FakeResponse:
    def __init__(self, payload, status_code=200, headers=None):
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


class _FakeSession:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []

    def get(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return next(self.responses)


class _FakeMarketDataSource:
    provider = "fake"

    def fetch_calendar(self, start: date, end: date) -> MarketDataPage:
        assert start == date(2024, 7, 3)
        assert end == date(2024, 7, 3)
        return MarketDataPage(
            request={"start": start.isoformat(), "end": end.isoformat()},
            payload=[
                {
                    "date": "2024-07-03",
                    "open": "09:30",
                    "close": "13:00",
                }
            ],
        )

    def iter_bar_pages(self, request: HistoricalBarsRequest):
        assert request.adjustment == "split"
        assert request.feed == "sip"
        assert request.timeframe == "1Min"
        yield MarketDataPage(
            request={"symbol": request.symbol, "page_token": None},
            payload={
                "bars": {
                    request.symbol: [
                        _bar("2024-07-03T13:29:00Z", 99.0),
                        _bar("2024-07-03T13:30:00Z", 100.0),
                        _bar("2024-07-03T16:59:00Z", 101.0),
                        _bar("2024-07-03T17:00:00Z", 102.0),
                    ]
                },
                "next_page_token": None,
            },
        )


def _bar(timestamp: str, close: float) -> dict:
    return {
        "t": timestamp,
        "o": close - 0.2,
        "h": close + 0.3,
        "l": close - 0.4,
        "c": close,
        "v": 1000,
        "n": 100,
        "vw": close - 0.05,
    }


@pytest.mark.parametrize("symbol", ["", "../AAPL", "AAPL/USD", "AAPL*"])
def test_snapshot_config_rejects_unsafe_stock_symbols(tmp_path: Path, symbol: str):
    with pytest.raises(ValueError, match="safe stock ticker"):
        SnapshotBuildConfig(
            snapshot_root=tmp_path / "snapshot",
            symbols=(symbol,),
            start=date(2024, 7, 3),
            end=date(2024, 7, 3),
        )


@pytest.mark.parametrize(
    "override",
    (
        {"schema_version": 3},
        {"adjustment": "raw"},
        {"timeframe": "5Min"},
        {"feed": ""},
        {"timezone_name": "Mars/Olympus_Mons"},
        {"usage_classification": ""},
    ),
)
def test_snapshot_config_rejects_unsupported_dataset_contracts(
    tmp_path: Path,
    override: dict,
):
    with pytest.raises(ValueError):
        SnapshotBuildConfig(
            snapshot_root=tmp_path / "snapshot",
            symbols=("AAPL",),
            start=date(2024, 7, 3),
            end=date(2024, 7, 3),
            **override,
        )


def test_snapshot_builder_writes_immutable_session_partitions_and_manifest(tmp_path: Path):
    output = tmp_path / "dataset_v2_test"
    result = build_market_snapshot(
        _FakeMarketDataSource(),
        SnapshotBuildConfig(
            snapshot_root=output,
            symbols=("AAPL",),
            start=date(2024, 7, 3),
            end=date(2024, 7, 3),
            feed="sip",
            adjustment="split",
        ),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 2
    assert manifest["provider"] == "fake"
    assert manifest["feed"] == "sip"
    assert manifest["adjustment"] == "split"
    assert manifest["session_counts"] == {
        "after_hours": 1,
        "pre_market": 1,
        "regular": 2,
    }
    assert manifest["row_count"] == 4
    assert manifest["symbols"] == ["AAPL"]
    assert manifest["content_sha256"]
    assert manifest["created_at_utc"].endswith("+00:00")
    assert manifest["usage_classification"] == "unverified_personal_internal_research"
    assert manifest["missing_bar_policy"] == "provider_omission_preserved_no_forward_fill"

    parquet_files = sorted((output / "bars").rglob("*.parquet"))
    assert len(parquet_files) == 3
    regular = pd.read_parquet(
        output / "bars",
        filters=[("session", "==", "regular")],
    ).sort_values("ts")
    assert regular["CLOSE"].tolist() == [100.0, 101.0]
    assert regular["session_id"].nunique() == 1
    assert str(regular["ts"].dt.tz) == "UTC"

    raw_files = sorted((output / "raw" / "fake").rglob("*.json.gz"))
    assert len(raw_files) == 2
    assert [record["path"] for record in manifest["raw_files"]] == [
        path.relative_to(output).as_posix() for path in raw_files
    ]
    assert all(len(record["sha256"]) == 64 for record in manifest["raw_files"])
    with gzip.open(raw_files[1], "rt", encoding="utf-8") as stream:
        raw_record = json.load(stream)
    assert raw_record["retrieved_at_utc"].endswith("+00:00")
    assert raw_record["request"]["symbol"] == "AAPL"
    assert raw_record["payload"]["bars"]["AAPL"][1]["c"] == 100.0

    with pytest.raises(FileExistsError, match="immutable"):
        build_market_snapshot(
            _FakeMarketDataSource(),
            SnapshotBuildConfig(
                snapshot_root=output,
                symbols=("AAPL",),
                start=date(2024, 7, 3),
                end=date(2024, 7, 3),
            ),
        )


def test_snapshot_content_hash_ignores_atomic_write_temporaries(tmp_path: Path):
    from itrader.inputs.market_data.snapshot import _content_hash

    (tmp_path / "bars.parquet").write_bytes(b"stable-content")
    expected = _content_hash(tmp_path)

    (tmp_path / "manifest.json.tmp").write_text(
        '{"interrupted": true}\n',
        encoding="utf-8",
    )

    assert _content_hash(tmp_path) == expected


def test_snapshot_catalog_queries_partitioned_parquet(tmp_path: Path):
    output = tmp_path / "dataset_v2_test"
    build_market_snapshot(
        _FakeMarketDataSource(),
        SnapshotBuildConfig(
            snapshot_root=output,
            symbols=("AAPL",),
            start=date(2024, 7, 3),
            end=date(2024, 7, 3),
        ),
    )

    with open_snapshot_catalog(output) as catalog:
        counts = catalog.execute(
            "SELECT session, COUNT(*) AS n FROM bars GROUP BY session ORDER BY session"
        ).fetchall()

    assert counts == [("after_hours", 1), ("pre_market", 1), ("regular", 2)]


def test_alpaca_source_authenticates_and_follows_page_tokens():
    session = _FakeSession(
        [
            _FakeResponse([{"date": "2024-01-02", "open": "09:30", "close": "16:00"}]),
            _FakeResponse({"bars": {"AAPL": [_bar("2024-01-02T14:30:00Z", 100)]}, "next_page_token": "next"}),
            _FakeResponse({"bars": {"AAPL": [_bar("2024-01-02T14:31:00Z", 101)]}, "next_page_token": None}),
        ]
    )
    source = AlpacaMarketDataSource("key-id", "secret", session=session)
    calendar = source.fetch_calendar(date(2024, 1, 2), date(2024, 1, 2))
    pages = list(
        source.iter_bar_pages(
            HistoricalBarsRequest(
                symbol="AAPL",
                start=pd.Timestamp("2024-01-02", tz="UTC").to_pydatetime(),
                end=pd.Timestamp("2024-01-03", tz="UTC").to_pydatetime(),
            )
        )
    )

    assert calendar.payload[0]["date"] == "2024-01-02"
    assert len(pages) == 2
    assert session.calls[0][0] == "https://paper-api.alpaca.markets/v2/calendar"
    assert session.calls[1][0] == "https://data.alpaca.markets/v2/stocks/bars"
    assert session.calls[1][1]["headers"] == {
        "APCA-API-KEY-ID": "key-id",
        "APCA-API-SECRET-KEY": "secret",
    }
    assert "page_token" not in session.calls[1][1]["params"]
    assert session.calls[2][1]["params"]["page_token"] == "next"


@pytest.mark.parametrize(
    "kwargs",
    ({"timeout_seconds": 0}, {"max_retries": -1}),
)
def test_alpaca_source_rejects_invalid_retry_configuration(kwargs):
    with pytest.raises(ValueError):
        AlpacaMarketDataSource("key-id", "secret", **kwargs)


def test_alpaca_source_falls_back_when_retry_after_is_not_numeric():
    delays = []
    session = _FakeSession(
        [
            _FakeResponse({}, status_code=429, headers={"Retry-After": "tomorrow"}),
            _FakeResponse([]),
        ]
    )
    source = AlpacaMarketDataSource(
        "key-id",
        "secret",
        session=session,
        max_retries=1,
        sleep=delays.append,
    )

    page = source.fetch_calendar(date(2024, 1, 2), date(2024, 1, 2))

    assert page.payload == []
    assert delays == [1.0]


def test_snapshot_builder_resumes_only_incomplete_symbol_years(tmp_path: Path):
    output = tmp_path / "dataset_v2_resume"

    class FailingSource(_FakeMarketDataSource):
        def iter_bar_pages(self, request: HistoricalBarsRequest):
            if request.symbol == "MSFT":
                raise RuntimeError("interrupted")
            yield from super().iter_bar_pages(request)

    with pytest.raises(RuntimeError, match="interrupted"):
        build_market_snapshot(
            FailingSource(),
            SnapshotBuildConfig(
                snapshot_root=output,
                symbols=("AAPL", "MSFT"),
                start=date(2024, 7, 3),
                end=date(2024, 7, 3),
            ),
        )

    class ResumeSource(_FakeMarketDataSource):
        def iter_bar_pages(self, request: HistoricalBarsRequest):
            assert request.symbol == "MSFT"
            yield from super().iter_bar_pages(request)

    result = build_market_snapshot(
        ResumeSource(),
        SnapshotBuildConfig(
            snapshot_root=output,
            symbols=("AAPL", "MSFT"),
            start=date(2024, 7, 3),
            end=date(2024, 7, 3),
        ),
    )

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["row_count"] == 8
    assert manifest["symbols"] == ["AAPL", "MSFT"]


def test_snapshot_builder_rebuilds_a_partially_written_symbol_year(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    output = tmp_path / "dataset_v2_partial_partition"
    original_write = snapshot_module._write_partitioned
    interrupted = False

    def write_regular_then_interrupt(frame, bars_root, part_index):
        nonlocal interrupted
        if not interrupted:
            interrupted = True
            original_write(frame.loc[frame["session"] == "regular"], bars_root, part_index)
            raise RuntimeError("interrupted after partial partition write")
        original_write(frame, bars_root, part_index)

    monkeypatch.setattr(snapshot_module, "_write_partitioned", write_regular_then_interrupt)
    config = SnapshotBuildConfig(
        snapshot_root=output,
        symbols=("AAPL",),
        start=date(2024, 7, 3),
        end=date(2024, 7, 3),
    )
    with pytest.raises(RuntimeError, match="partial partition"):
        build_market_snapshot(_FakeMarketDataSource(), config)

    monkeypatch.setattr(snapshot_module, "_write_partitioned", original_write)
    result = build_market_snapshot(_FakeMarketDataSource(), config)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["session_counts"] == {
        "after_hours": 1,
        "pre_market": 1,
        "regular": 2,
    }
    assert manifest["row_count"] == 4
