"""CLI entry points for the training pipeline."""

from __future__ import annotations

import argparse
import sys


def main() -> int:
    parser = argparse.ArgumentParser(prog="progno-train")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("update_data", help="Fetch latest Sackmann data")
    sub.add_parser("ingest", help="Ingest CSVs to staging parquet")
    sub.add_parser("elo", help="Compute Elo state snapshot")
    publish = sub.add_parser("publish", help="Publish artifacts to app-data")
    publish.add_argument("version")

    args = parser.parse_args()
    if args.command == "update_data":
        print("not implemented yet")
        return 1
    if args.command == "ingest":
        print("not implemented yet")
        return 1
    if args.command == "elo":
        print("not implemented yet")
        return 1
    if args.command == "publish":
        print(f"not implemented yet (version={args.version})")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
