from datetime import datetime

class Timer:
    def __init__(self):
        self._start: None | datetime = None

    def __enter__(self) -> None:
        self._start = datetime.now()

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        assert self._start is not None
        stop = datetime.now()
        print(f"Total run time {stop - self._start}")
