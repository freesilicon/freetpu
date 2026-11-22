"""CLI entrypoint for `python3 -m freetpu_tile --self-test`."""

from __future__ import annotations

import argparse
import sys
import unittest
from pathlib import Path


def run_self_test(verbosity: int = 2) -> int:
    tests_dir = Path(__file__).resolve().parent / "tests"
    suite = unittest.defaultTestLoader.discover(start_dir=str(tests_dir), pattern="test_*.py")
    result = unittest.TextTestRunner(verbosity=verbosity).run(suite)
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="freetpu_tile", description="FreeTPU Tile DSL CLI & self-test runner")
    ap.add_argument("--self-test", action="store_true", help="Run offline unit & parity test suite")
    ap.add_argument("-q", "--quiet", action="store_true", help="Quiet test output")
    args = ap.parse_args(argv)

    if args.self_test:
        return run_self_test(verbosity=1 if args.quiet else 2)
    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
