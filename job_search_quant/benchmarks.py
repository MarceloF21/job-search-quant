"""Toronto wage benchmarks. Update them when the sources publish new figures."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Benchmark:
    name: str
    hourly: float
    source: str
    as_of: str


ONTARIO_MINIMUM_WAGE = Benchmark(
    "Ontario minimum wage", 17.95, "https://www.ontario.ca/minimumwage", "2026-10-01")
GTA_LIVING_WAGE = Benchmark(
    "GTA living wage", 27.20, "Ontario Living Wage Network, 2025 rates", "2025-11")

DEFAULT_BENCHMARKS = (ONTARIO_MINIMUM_WAGE, GTA_LIVING_WAGE)
