"""Small command-line utilities for installation diagnostics."""

from __future__ import annotations

import argparse
import platform
import sys

from . import __version__
from .epipvr import EpiPvrBackend


def _doctor() -> int:
    print(f"Cropmix {__version__}")
    print(f"Python {platform.python_version()} ({sys.executable})")
    backend = EpiPvrBackend()
    status = backend.check_installation()
    print(f"Rscript: {'OK' if status['rscript'] else 'NOT FOUND'}")
    if status["rscript"]:
        print(f"EpiPvr: {status.get('version') or 'NOT INSTALLED'}")
    if status.get("message"):
        print(status["message"])
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="cropmix", description="Cropmix utilities")
    parser.add_argument("--version", action="store_true", help="Print Cropmix version")
    subparsers = parser.add_subparsers(dest="command")
    subparsers.add_parser("doctor", help="Check Python, Rscript and EpiPvr availability")
    args = parser.parse_args(argv)
    if args.version:
        print(__version__)
        return 0
    if args.command == "doctor":
        return _doctor()
    parser.print_help()
    return 0
