from __future__ import annotations
from ..solver import Solver
import time
from datetime import timedelta
import multiprocessing as mp
from multiprocessing.synchronize import Lock
from multiprocessing.sharedctypes import Synchronized, Value
from concurrent.futures import ProcessPoolExecutor
import typing as tp
from ase import Atoms

class Parallel:
    class ProgressReporter(Solver.Callable):
        lock: Lock | None = None
        shared_steps: Synchronized[int] | None = None

        @classmethod
        def init_progress(
            cls,
            shared_lock: Lock,
            shared_counter: Synchronized[int],
        ) -> None:
            cls.lock = shared_lock
            cls.shared_steps = shared_counter

        def __init__(self, every: int, total: int) -> None:
            self.every = every
            self.total = total
            self.last_step = 0
            self.start = time.perf_counter()

        def matches(self, step: int) -> bool:
            return step % self.every == 0

        def next_match(self, step: int) -> int:
            return self.every - step % self.every

        def call(self, step: int, atoms: Atoms) -> None:
            assert isinstance(self.lock, Lock), \
                "Reporter is not initialized"
            assert isinstance(self.shared_steps, Synchronized), \
                "Reporter is not initialized"

            with self.lock:
                delta = max(0, step - self.last_step)
                self.shared_steps.value += delta
                self.last_step = step

                completed = self.shared_steps.value
                fraction = min(completed / self.total, 1.0)

                elapsed = time.perf_counter() - self.start
                eta_seconds = (
                    elapsed * (1 - fraction) / fraction
                    if fraction > 0 else 0
                )

                eta = timedelta(seconds=eta_seconds)
                hours, remainder = divmod(int(eta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)

                print(
                    f"Overall progress: {fraction * 100:6.2f}% "
                    f"({completed:,}/{self.total:,} steps) "
                    f"ETA {hours:02d}:{minutes:02d}:{seconds:02d}",
                    end="          \r",
                    flush=True,
                )

    class ProgressFlipReporter:
        lock: Lock | None = None
        shared_steps: Synchronized[int] | None = None
        start: float | None = None

        @classmethod
        def init_progress(
            cls,
            shared_lock: Lock,
            shared_counter: Synchronized[int],
        ) -> None:
            cls.lock = shared_lock
            cls.shared_steps = shared_counter
            cls.start = time.perf_counter()

        def __init__(self, total: int) -> None:
            self.total = total
            self.done = False

        def flip(self) -> None:
            self.done = True

            assert self.lock is not None
            assert self.shared_steps is not None
            assert self.start is not None

            with self.lock:
                delta = 1
                self.shared_steps.value += delta

                completed = self.shared_steps.value
                fraction = min(completed / self.total, 1.0)

                elapsed = time.perf_counter() - self.start
                eta_seconds = (
                    elapsed * (1 - fraction) / fraction
                    if fraction > 0 else 0
                )

                eta = timedelta(seconds=eta_seconds)
                hours, remainder = divmod(int(eta.total_seconds()), 3600)
                minutes, seconds = divmod(remainder, 60)

                print(
                    f"Overall progress: {fraction * 100:6.2f}% "
                    f"({completed:,}/{self.total:,} iterations) "
                    f"ETA {hours:02d}:{minutes:02d}:{seconds:02d}",
                    end="          \r",
                    flush=True,
                )

    def __init__(
        self,
        parallel: int,
    ) -> None:
        self._parallel = parallel
        self._context = mp.get_context("spawn")
        self._init_args = (
            self._context.Lock(),
            self._context.Value("q", 0),
            self._context.Lock(),
            self._context.Value("q", 0),
        )

    @staticmethod
    def _pool_initializer(
        lock: Lock,
        steps: Synchronized[int],
        lock2: Lock,
        steps2: Synchronized[int],
    ) -> None:
        Parallel.ProgressReporter.init_progress(lock, steps)
        Parallel.ProgressFlipReporter.init_progress(lock2, steps2)

    def progress_reporter_instances(
        self,
        every: int,
        total: int,
        number: int,
    ) -> tp.Iterable[Parallel.ProgressReporter]:
        for i in range(number):
            yield Parallel.ProgressReporter(every, total)

    def progress_flip_reporter_instances(
        self,
        number: int,
    ) -> tp.Iterable[Parallel.ProgressFlipReporter]:
        for i in range(number):
            yield Parallel.ProgressFlipReporter(number)

    def run(
        self,
        function: tp.Callable[..., tp.Any],
        *arguments: tp.Iterable[tp.Any],
    ) -> tp.Iterable[tp.Any]:
        with ProcessPoolExecutor(
            mp_context=self._context,
            initializer=self._pool_initializer,
            initargs=self._init_args,
            max_workers=self._parallel,
        ) as executor:
            return executor.map(
                function,
                *arguments,
            )
