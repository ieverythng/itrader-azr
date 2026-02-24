from __future__ import annotations

import argparse
import runpy
import sys


COMMAND_TO_MODULE = {
    "train": "itrader.rl_lib.train.train_solo_V2",
    "healthcheck": "itrader.tools.healthcheck",
    "ablation": "itrader.scripts.experiments.run_ablation",
    "gen-task-pool": "itrader.scripts.gen_task_pool",
}


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m itrader",
        description="iTrader command gateway",
    )
    p.add_argument(
        "command",
        nargs="?",
        default="help",
        help=f"Command to run. Available: {', '.join(sorted(COMMAND_TO_MODULE.keys()))}",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, remainder = parser.parse_known_args(argv)
    command = str(args.command).strip().lower()

    if command in {"help", "-h", "--help"}:
        parser.print_help()
        return 0

    module = COMMAND_TO_MODULE.get(command)
    if module is None:
        parser.print_help()
        print(f"\nUnknown command: {command}")
        return 2

    sys.argv = [f"python -m {module}", *remainder]
    runpy.run_module(module, run_name="__main__")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
