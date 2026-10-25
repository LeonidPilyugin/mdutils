from __future__ import annotations

import multiprocessing as mp
import random
import time
import typing as tp
from abc import ABC, abstractmethod
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime, timedelta
from multiprocessing.managers import SyncManager, ValueProxy
from types import TracebackType
from typing import ParamSpec, Self, TypeVar

from ase import Atoms

from ..solver import Solver


class ProgressGuard:
    def __init__(self, lock: AbstractContextManager[object], steps: ValueProxy[int]) -> None:
        self._lock = lock
        self._shared_steps = steps

    def increment(self, done: int) -> None:
        assert self._lock is not None
        assert self._shared_steps is not None

        with self._lock:
            self._shared_steps.value += done

    def value(self) -> int:
        assert self._lock is not None
        assert self._shared_steps is not None

        with self._lock:
            return self._shared_steps.value

class ProgressCallable(Solver.Callable):
    def __init__(
        self,
        guard: ProgressGuard,
        every: int,
    ) -> None:
        self._guard = guard
        self._last_step = 0
        self._every = every

    def call(self, step: int, atoms: Atoms) -> None:
        self._guard.increment(step - self._last_step)
        self._last_step = step

    def matches(self, step: int) -> bool:
        return step % self._every == 0

    def next_match(self, step: int) -> int:
        return self._every - step % self._every


class ElapsedEstimator(ABC):
    def __init__(self) -> None:
        self._total: None | int = None

    def reset(self, total: int) -> None:
        self._total = total

    @property
    def total(self) -> int:
        assert self._total is not None
        return self._total

    @abstractmethod
    def estimate(self, done: int, now: datetime) -> None | datetime:
        pass


class ExponentialSmoothingEstimator(ElapsedEstimator):
    def __init__(self, *, alpha: float = 0.9, capacity = 10) -> None:
        super().__init__()
        assert 0.0 < alpha <= 1.0
        assert(capacity > 1)
        self._alpha = alpha
        self._performance: float | None = None
        self._done_buffer: deque[int] = deque()
        self._time_buffer: deque[datetime] = deque()
        self._capacity = capacity

    def estimate(self, done: int, now: datetime) -> datetime | None:
        if done not in self._done_buffer:
            self._done_buffer.append(done)
            self._time_buffer.append(now)

        assert len(self._done_buffer) == len(self._time_buffer)

        if len(self._time_buffer) < 2:
            return None

        while len(self._time_buffer) > self._capacity:
            self._time_buffer.popleft()
            self._done_buffer.popleft()

        cur_perf = ((self._done_buffer[-1] - self._done_buffer[0]) /
                    (self._time_buffer[-1] - self._time_buffer[0]).total_seconds())

        self._performance =  cur_perf if self._performance is None else \
            self._performance * self._alpha + cur_perf * (1.0 - self._alpha)

        if self._performance == 0.0:
            return None

        todo = self.total - done

        est = self._time_buffer[-1] + timedelta(seconds=todo / self._performance)
        # print(est, now)
        return est


class ProgressReporter(ABC):
    @dataclass
    class Data:
        total: int
        done: int
        start: datetime
        end: datetime | None
        now: datetime

        @property
        def todo(self) -> int:
            return self.total - self.done

        @property
        def left(self) -> timedelta | None:
            if self.end is None:
                return None
            return self.end - self.now

        @property
        def passed(self) -> timedelta:
            return self.now - self.start

        @property
        def part(self) -> float:
            return self.done / self.total

    @abstractmethod
    def report(self, data: Data) -> None:
        pass


class DefaultProgressReporter(ProgressReporter):
    def __init__(self) -> None:
        pass

    def report(self, data: ProgressReporter.Data) -> None:
        def format_timedelta(td: timedelta | None) -> str:
            if td is not None:
                hours, remainder = divmod(int(td.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)
                return f"{hours}:{minutes:02}:{seconds:02}"
            return ""

        message = (f"Done {data.done} of {data.total} iterations "
            f"({data.part * 100:.2f} %)")

        if data.end is not None:
            message += "; expected finish time: "
            message += f"{data.end.strftime('%Y-%m-%d %H:%M')} "
            message += f"({format_timedelta(data.left)} left)"

        print(f"\r\033[K{message}", end="", flush=True)

P = ParamSpec("P")
R = TypeVar("R")

class Worker[**P, R](ABC):
    def __init__(self, *, guard: ProgressGuard) -> None:
        self._guard = guard

    def update(self, done: int = 1) -> None:
        self._guard.increment(done)

    @abstractmethod
    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        pass

    def total(self) -> int:
        return 1


class Parallel:
    def __init__(
        self,
        parallel: int,
        estimator: ElapsedEstimator | None = None,
        reporter: ProgressReporter | None = None,
        delay: float = 0.1,
    ) -> None:
        self._parallel = parallel
        self._estimator = estimator or ExponentialSmoothingEstimator(
            capacity=parallel * 2,
        )
        self._reporter = reporter or DefaultProgressReporter()
        self._guard: None | ProgressGuard = None
        self._manager: SyncManager | None = None
        self._delay = delay

    @property
    def guard(self) -> ProgressGuard:
        assert self._guard is not None
        return self._guard

    def __enter__(self) -> Self:
        self._manager = mp.Manager()

        assert self._manager is not None

        self._guard = ProgressGuard(
            self._manager.Lock(),
            self._manager.Value("q", 0)
        )

        self._executor_cm = ProcessPoolExecutor(
            max_workers=self._parallel,
            mp_context=mp.get_context("spawn"),
        )
        self.executor = self._executor_cm.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> bool | None:
        assert self._manager is not None
        try:
            return self._executor_cm.__exit__(
                exc_type, exc_value, traceback
            )
        finally:
            self._manager.shutdown()

    def map(
        self,
        worker: Worker,
        *arguments: tp.Iterable[tp.Any],
    ) -> list[tp.Any]:
        assert self._guard is not None

        futures = [
            self.executor.submit(worker, *item)
            for item in zip(*arguments)
        ]

        total = worker.total() * len(futures)
        self._estimator.reset(total)
        start = datetime.now()

        while not all(f.done() for f in futures):
            time.sleep(self._delay)
            done = self._guard.value()
            now = datetime.now()
            finish = self._estimator.estimate(done, now)

            if finish is not None:
                self._reporter.report(ProgressReporter.Data(
                    total=total,
                    done=done,
                    start=start,
                    end=finish,
                    now=now,
                ))
        print()

        return [f.result() for f in futures]

class _DummyWorker(Worker[[int], None]):
    def __call__(self, seed: int) -> None:
        rng = random.Random()
        rng.seed(seed)
        for i in range(10):
            time.sleep(rng.randint(1, 5))
            self.update()

    def total(self) -> int:
        return 10

if __name__ == "__main__":

    with Parallel(parallel=4, delay=0.2) as parallel:
        worker = _DummyWorker(guard=parallel.guard)
        results = list(parallel.map(worker, range(50)))
