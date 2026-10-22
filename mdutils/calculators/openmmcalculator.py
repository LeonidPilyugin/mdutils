import numpy as np
import openmm as mm
from openmm import unit
from ase.calculators.calculator import Calculator, all_changes
from ase import units as ase_units
from ase import Atoms
import typing as tp
import gc
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

        cell = openmmhelper.Cell.from_ase(atoms.cell.array)
        positions = openmmhelper.Positions.from_ase(atoms.get_positions())
        types = atoms.get_chemical_symbols()

        system = mm.System()
        system.setDefaultPeriodicBoxVectors(*cell)

        for _ in positions:
            system.addParticle(0)

        for i, (force, data) in enumerate(zip(self._forces, self._forcedata)):
            if hasattr(force, "addParticle"):
                for i, _ in enumerate(positions):
                    force.addParticle(*data[types[i]])

            system.addForce(force)

        self._context = mm.Context(
            system,
            self._integrator,
            self._platform,
            self._platform_prop
        )

        self._context.setPositions(positions)

    def calculate(
        self,
        atoms: None | Atoms = None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        assert isinstance(atoms, Atoms)

        self._context.setPositions(
            openmmhelper.Positions.from_ase(atoms.get_positions())
        )

        self._context.setPeriodicBoxVectors(
            *openmmhelper.Cell.from_ase(np.array(atoms.cell))
        )

        state = self._context.getState(
            getEnergy=True,
            getForces=True,
        )

        self.results["energy"] = openmmhelper.Energy.from_openmm(
            state.getPotentialEnergy()
        )
        self.results["forces"] = openmmhelper.Forces.from_openmm(
            state.getForces(asNumpy=True)
        )

    def close(self):
        del self._context
        del self._integrator
        gc.collect()

