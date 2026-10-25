import ase
import numpy as np
import numpy.typing as npt
import openmm as mm
from openmm.unit import Quantity


class Cell:
    def __init__(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_ase(cell: npt.ArrayLike) -> Quantity:
        return np.asarray(cell, dtype=np.float64) * 0.1 * mm.unit.nanometer # pyright: ignore[reportAttributeAccessIssue]

class Positions:
    def __init__(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_ase(positions: npt.ArrayLike) -> Quantity:
        return np.asarray(positions) * 0.1 * mm.unit.nanometer # pyright: ignore[reportAttributeAccessIssue]

    @staticmethod
    def from_openmm(positions: Quantity) -> npt.NDArray[np.float64]:
        return np.asarray(
            positions.value_in_unit(mm.unit.nanometer), # pyright: ignore[reportAttributeAccessIssue]
            dtype=np.float64
        ) * 10.0

class Velocities:
    def __init__(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_ase(velocities: npt.ArrayLike) -> Quantity:
        return (
            np.asarray(velocities)
            * 100.0
            * ase.units.fs          # pyright: ignore[reportAttributeAccessIssue]
            * mm.unit.nanometer     # pyright: ignore[reportAttributeAccessIssue]
            / mm.unit.picosecond    # pyright: ignore[reportAttributeAccessIssue]
        )

    @staticmethod
    def from_openmm(velocities: Quantity) -> npt.NDArray[np.float64]:
        return np.asarray(
            velocities.value_in_unit(
                mm.unit.nanometer / mm.unit.picosecond # pyright: ignore[reportAttributeAccessIssue]
            ),
            dtype=np.float64
        ) / 100.0 / ase.units.fs # pyright: ignore[reportAttributeAccessIssue]

class Energy:
    def __init__(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_openmm(energy: Quantity) -> float:
        return float(energy.value_in_unit(
            mm.unit.kilojoule_per_mole
        ) * ase.units.kJ / ase.units.mol) # pyright: ignore[reportAttributeAccessIssue]

class Forces:
    def __init__(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_openmm(forces: Quantity) -> npt.NDArray[np.float64]:
        return np.asarray(
            forces.value_in_unit(
                mm.unit.kilojoule_per_mole / mm.unit.nanometer # pyright: ignore[reportAttributeAccessIssue]
            ),
            dtype=np.float64,
        ) * ase.units.kJ / ase.units.mol / 10.0 # pyright: ignore[reportAttributeAccessIssue]

class Masses:
    def __init__(self) -> None:
        raise NotImplementedError()

    @staticmethod
    def from_ase(masses: npt.ArrayLike) -> Quantity:
        return np.asarray(masses) * mm.unit.amu
