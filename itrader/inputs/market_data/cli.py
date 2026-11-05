from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from .alpaca import AlpacaMarketDataSource
from .contracts import SnapshotBuildConfig
from .snapshot import build_market_snapshot


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"Environment file not found: {path}")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if not separator or not key.strip():
            raise ValueError(f"Malformed environment assignment in {path}")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def _parse_date(value: str) -> date:
    return date.fromisoformat(value)


def _build(args: argparse.Namespace) -> int:
    _load_env_file(Path(args.env_file).expanduser().resolve())
    key_id = os.environ.get("APCA_API_KEY_ID", "")
    secret = os.environ.get("APCA_API_SECRET_KEY", "")
    source = AlpacaMarketDataSource(key_id, secret)
    result = build_market_snapshot(
        source,
        SnapshotBuildConfig(
            snapshot_root=Path(args.output),
            symbols=tuple(value.strip() for value in args.symbols.split(",") if value.strip()),
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            feed=args.feed,
            adjustment="split",
            usage_classification=args.usage_classification,
        ),
        progress=lambda message: print(message, flush=True),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    print(json.dumps(manifest, indent=2, sort_keys=True), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build immutable iTrader market-data snapshots")
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build", help="Build an Alpaca historical-bars snapshot")
    build.add_argument("--output", required=True)
    build.add_argument("--symbols", required=True)
    build.add_argument("--start", required=True)
    build.add_argument("--end", required=True)
    build.add_argument("--feed", default="sip", choices=["sip", "iex"])
    build.add_argument(
        "--usage-classification",
        default="unverified_personal_internal_research",
    )
    build.add_argument("--env-file", default=".env")
    build.set_defaults(handler=_build)
    args = parser.parse_args()
    return int(args.handler(args))


if __name__ == "__main__":
    raise SystemExit(main())
