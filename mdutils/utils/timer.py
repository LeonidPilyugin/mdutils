from datetime import datetime
from types import TracebackType


class Timer:
    def __init__(self) -> None:
        self._start: None | datetime = None

    def __enter__(self) -> None:
        self._start = datetime.now()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        assert self._start is not None
        stop = datetime.now()
        print(f"Total run time {stop - self._start}")
