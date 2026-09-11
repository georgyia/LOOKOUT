"""What a run cost, per stage.

Coverage says how much of a recording survived each stage and diagnostics say
whether the result is worth reading. Neither says what producing it cost, so
there has been no way to answer whether a long recording is minutes or hours of
work, nor which stage to attack if the answer is unacceptable.

Timing is recorded the way coverage is: measured during the run, carried in the
record, rendered in the report. It belongs with provenance rather than with
results, because a duration is meaningless without the machine, the
configuration and the adapters that produced it.

Nothing here asserts a threshold. Wall clock on a shared runner varies by more
than the effects worth catching, so regressions are guarded by asserting the
*work* done — call counts and complexity — and durations are recorded for humans
to compare.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

__all__ = ["StageTiming", "RunTiming", "Stopwatch"]


@dataclass(frozen=True)
class StageTiming:
    """Total wall time and call count for one stage."""

    stage: str
    seconds: float
    calls: int

    def share_of(self, total: float) -> float:
        return self.seconds / total if total > 0 else 0.0


@dataclass(frozen=True)
class RunTiming:
    """What a run cost, and what that implies for a longer recording."""

    stages: tuple[StageTiming, ...] = ()
    wall_seconds: float = 0.0
    video_seconds: float = 0.0

    @property
    def realtime_factor(self) -> float:
        """Seconds of video processed per second of wall time.

        The number a user actually reasons in: 0.8 means a 43-minute recording
        takes about 54 minutes. A per-stage duration does not answer that.
        """

        return self.video_seconds / self.wall_seconds if self.wall_seconds > 0 else 0.0

    def projected_seconds(self, video_seconds: float) -> float:
        """Wall time this run's throughput implies for a recording of that length."""

        factor = self.realtime_factor
        return video_seconds / factor if factor > 0 else 0.0

    @property
    def dominant(self) -> StageTiming | None:
        """The stage worth attacking first."""

        return max(self.stages, key=lambda s: s.seconds) if self.stages else None


class Stopwatch:
    """Accumulates wall time per stage.

    A stopwatch that is never used costs nothing and reports nothing, so the
    pipeline can be instrumented unconditionally.
    """

    def __init__(self) -> None:
        self._seconds: dict[str, float] = {}
        self._calls: dict[str, int] = {}
        self._started = time.perf_counter()

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            elapsed = time.perf_counter() - start
            self._seconds[name] = self._seconds.get(name, 0.0) + elapsed
            self._calls[name] = self._calls.get(name, 0) + 1

    def calls(self, name: str) -> int:
        """How many times a stage ran. The stable thing to assert in a test."""

        return self._calls.get(name, 0)

    def result(self, video_seconds: float = 0.0) -> RunTiming:
        return RunTiming(
            stages=tuple(
                StageTiming(
                    stage=name,
                    seconds=round(self._seconds[name], 6),
                    calls=self._calls[name],
                )
                for name in sorted(self._seconds, key=lambda n: -self._seconds[n])
            ),
            wall_seconds=round(time.perf_counter() - self._started, 6),
            video_seconds=round(video_seconds, 3),
        )
