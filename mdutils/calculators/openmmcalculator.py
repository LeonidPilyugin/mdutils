import numpy as np
import openmm as mm
from openmm import unit
from ..solvers import OpenMmSolver
from ase.calculators.calculator import Calculator, all_changes
from ase import units as ase_units
from ase import Atoms
import typing as tp
from .. import openmmhelper


class OpenMmCalculator(Calculator):

    implemented_properties = ["energy", "forces"]

    def __init__(
        self,
        atoms: Atoms,
        integrator: mm.Integrator,
        platform: mm.Platform,
        platform_prop: tp.Dict[str, str],
        forces: tp.Iterable[mm.Force],
        forcedata: tp.Iterable[tp.Dict[str, tp.Any]],
        **kwargs,
    ):
        super().__init__(**kwargs)

        self._integrator = integrator
        self._platform = platform
        self._platform_prop = platform_prop
        self._forces = forces
        self._forcedata = forcedata
        self._atoms = atoms

        self._solver = None

    def __enter__(self):
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

    def __exit__(self, exc_type, exc_value, traceback):
        self._solver_cm.__exit__(
            exc_type, exc_value, traceback
        )

    def calculate(
        self,
        atoms: None | Atoms = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        assert isinstance(atoms, Atoms)
        assert self._solver is not None

        self._solver.update(atoms)

        state = self._solver.get_state(
            getEnergy=True,
            getForces=True,
        )

        self.results["energy"] = openmmhelper.Energy.from_openmm(
            state.getPotentialEnergy()
        )
        self.results["forces"] = openmmhelper.Forces.from_openmm(
            state.getForces(asNumpy=True)
        )

