from abc import ABC, abstractmethod
from ase import Atoms

class System(ABC):
    @abstractmethod
    def create_system(self) -> Atoms:
        pass

