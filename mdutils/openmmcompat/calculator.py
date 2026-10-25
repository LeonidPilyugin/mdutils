from __future__ import annotations

import typing as tp
from types import TracebackType
from typing import Self

import openmm as mm
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes

from . import helper as openmmhelper
from .solver import OpenMmSolver


class OpenMmCalculator(Calculator):

    implemented_properties: tp.ClassVar = ["energy", "forces"] # pyright: ignore[reportIncompatibleVariableOverride]

    def __init__(
        self,
        atoms: Atoms,
        integrator: mm.Integrator,
        platform: mm.Platform,
        platform_prop: dict[str, str],
        forces: tp.Iterable[mm.Force],
        forcedata: tp.Iterable[dict[str, tp.Any]],
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)

        self._integrator = integrator
        self._platform = platform
        self._platform_prop = platform_prop
        self._forces = forces
        self._forcedata = forcedata
        self._atoms = atoms

        self._solver = None

    def __enter__(self) -> Self:
        self._solver_cm = OpenMmSolver(
            self._integrator,
            self._platform,
            self._platform_prop,
            self._forces,
            self._forcedata
        )

        self._solver = self._solver_cm.__enter__()
        self._solver.load_system(self._atoms)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        self._solver_cm.__exit__(
            exc_type, exc_value, traceback
        )

    def calculate(
        self,
        atoms: None | Atoms = None,
        properties: tp.Sequence[str] = ("energy", "forces"),
        system_changes: tp.Sequence[str] = all_changes,
    ) -> None:
        super().calculate(atoms, properties, system_changes)

        assert isinstance(atoms, Atoms)
        assert self._solver is not None

        self._solver.update(atoms)

        state = self._solver.get_state(
            getEnergy=True,
            getForces=True,
        )

        self.results["energy"] = openmmhelper.Energy.from_openmm(
            state.getPotentialEnergy() # pyright: ignore[reportAttributeAccessIssue]
        )
        self.results["forces"] = openmmhelper.Forces.from_openmm(
            state.getForces(asNumpy=True) # pyright: ignore[reportAttributeAccessIssue]
        )

