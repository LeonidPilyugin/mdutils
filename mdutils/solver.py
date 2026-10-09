from abc import ABC, abstractmethod
from ase import Atoms

class Solver(ABC):
    class Callable(ABC):
        @abstractmethod
        def call(self, step: int, atoms: Atoms) -> None:
            pass

        @abstractmethod
        def matches(self, step: int) -> bool:
            pass

        @abstractmethod
        def next_match(self, step: int) -> int:
            pass

    def __init__(self) -> None:
        self.callables = []

    @abstractmethod
    def load_system(self, atoms: Atoms) -> None:
        pass

    def attach(self, callable_) -> None:
        self.callables.append(callable_)

    @abstractmethod
    def run(self, steps: int) -> None:
        pass
