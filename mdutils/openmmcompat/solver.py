from __future__ import annotations

import gc
import typing as tp
from types import TracebackType
from typing import Self

import numpy as np
import openmm as mm
from ase import Atoms
from ase.constraints import FixAtoms

from ..solver import Solver
from . import helper as openmmhelper


class OpenMmSolver(Solver):
    def __init__(
        self,
        integrator: mm.Integrator,
        platform: mm.Platform,
        platform_prop: dict[str, str],
        forces: tp.Iterable[mm.Force],
        forcedata: tp.Iterable[dict[str, tp.Any]],
    ) -> None:
        super().__init__()
        self._integrator = integrator
        self._platform = platform
        self._forces = forces
        self._forcedata = forcedata
        self._platform_prop = platform_prop
        self._context: None | mm.Context = None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None
    ) -> None:
        self.close()


    def load_system(self, atoms: Atoms) -> None:
        self._atoms = atoms.copy()

        masses = openmmhelper.Masses.from_ase(np.array(atoms.get_masses()))
        types = atoms.get_chemical_symbols()

        system = mm.System()

        system.setDefaultPeriodicBoxVectors(
            *openmmhelper.Cell.from_ase(np.array(atoms.cell))
        )

        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                masses[constraint.get_indices()] = (
                    openmmhelper.Masses.from_ase(0.0)
                )

        for mass in masses:
            system.addParticle(mass)

        for force, data in zip(self._forces, self._forcedata):
            if hasattr(force, "addParticle"):
                for i, _ in enumerate(masses):
                    force.addParticle(*data[types[i]]) # pyright: ignore[reportAttributeAccessIssue]

            system.addForce(force)

        self._context = mm.Context(
            system,
            self._integrator,
            self._platform,
            self._platform_prop,
        )

        self.update(atoms)

    def update(self, atoms: Atoms) -> None:
        assert self._context is not None

        self._context.setPositions(
            openmmhelper.Positions.from_ase(np.array(atoms.positions))
        )
        self._context.setVelocities(
            openmmhelper.Velocities.from_ase(atoms.get_velocities())
        )
        self._context.setPeriodicBoxVectors(
            *openmmhelper.Cell.from_ase(np.array(atoms.cell))
        )

    def get_atoms(self) -> Atoms:
        from ase.calculators.calculator import Calculator, all_changes

        class _Calculator(Calculator):
            implemented_properties: tp.ClassVar = ['energy', 'forces'] # pyright: ignore[reportIncompatibleVariableOverride]

            def __init__(self, energy, forces=None, **kwargs):
                super().__init__(**kwargs)
                self.energy = energy
                self.forces = forces

            def calculate(self, atoms=None, properties=('energy',),
                          system_changes=all_changes):
                super().calculate(atoms, properties, system_changes)
                self.results['energy'] = self.energy
                if self.forces is not None:
                    self.results['forces'] = self.forces

        state = self.get_state(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=False,
            getForces=True,
            getEnergy=True,
        )

        atoms = self._atoms.copy();

        atoms.set_positions(
            openmmhelper.Positions.from_openmm(
                state.getPositions(asNumpy=True) # pyright: ignore[reportAttributeAccessIssue]
            )
        )

        atoms.set_velocities(
            openmmhelper.Velocities.from_openmm(
                state.getVelocities(asNumpy=True) # pyright: ignore[reportAttributeAccessIssue]
            )
        )

        atoms.calc = _Calculator(
            energy=openmmhelper.Energy.from_openmm(
                energy=state.getPotentialEnergy() # pyright: ignore[reportAttributeAccessIssue]
            ),
            forces=openmmhelper.Forces.from_openmm(
                state.getForces(asNumpy=True) # pyright: ignore[reportAttributeAccessIssue]
            ),
        )

        return atoms

    def run(self, steps: int) -> None:
        steps_done = 0

        while steps_done < steps:
            to_run = min(
                x.next_match(steps_done) for x in self.callables
            )

            self._integrator.step(to_run)
            steps_done += to_run

            atoms = self.get_atoms()

            for c in self.callables:
                if c.matches(steps_done):
                    c.call(steps_done, atoms)

    def get_state(self, **kwargs) -> mm.Context:
        assert self._context is not None
        return self._context.getState(**kwargs)

    def minimize(self, **kwargs) -> None:
        assert self._context is not None
        mm.LocalEnergyMinimizer.minimize(
            self._context,
            **kwargs
        )

    def close(self):
        del self._context
        del self._integrator
        gc.collect()
