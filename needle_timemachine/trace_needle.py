"""Compatibility wrapper for the package trace_needle CLI."""
from __future__ import annotations

from src.needle_timemachine.trace_needle import main

if __name__ == "__main__":
    raise SystemExit(main())
