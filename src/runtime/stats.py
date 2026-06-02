from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator


@dataclass
class RuntimeStats:
    stage_totals: dict[str, float] = field(default_factory=dict)
    frame_count: int = 0
    start_time: float = field(default_factory=time.time)

    @contextmanager
    def measure(self, stage: str) -> Iterator[list[float]]:
        start = time.time()
        elapsed_ref = [0.0]
        try:
            yield elapsed_ref
        finally:
            elapsed = time.time() - start
            elapsed_ref[0] = elapsed
            self.add(stage, elapsed)

    def add(self, stage: str, elapsed_s: float) -> None:
        self.stage_totals[stage] = self.stage_totals.get(stage, 0.0) + elapsed_s

    def increment_frame(self) -> None:
        self.frame_count += 1

    def elapsed_s(self) -> float:
        return time.time() - self.start_time

    def average_ms(self, stage: str) -> float:
        if self.frame_count <= 0:
            return 0.0
        return self.stage_totals.get(stage, 0.0) / self.frame_count * 1000.0

    def average_fps(self) -> float:
        elapsed = self.elapsed_s()
        return self.frame_count / elapsed if elapsed > 0 else 0.0


def pct(part_ms: float, total_ms: float) -> float:
    return part_ms / total_ms * 100.0 if total_ms > 0 else 0.0
