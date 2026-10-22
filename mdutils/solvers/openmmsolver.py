import typing as tp
from dataclasses import dataclass
import openmm as mm
import numpy as np
from ase import Atoms
from ase import units as ase_units
from ase.constraints import FixAtoms
from ase.units import fs
import gc
from ..solver import Solver
from .. import openmmhelper

class OpenMmSolver(Solver):
    def __init__(
        self,
        integrator: mm.Integrator,
        platform: mm.Platform,
        platform_prop: tp.Dict[str, str],
        forces: tp.Iterable[mm.Force],
        forcedata: tp.Iterable[tp.Dict[str, tp.Any]],
    ) -> None:
        super().__init__()
        self._integrator = integrator
        self._platform = platform
        self._forces = []
        self._forcedata = forcedata
        self._platform_prop = platform_prop
        self._context: None | mm.Context = None


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
                    force.addParticle(*data[types[i]])

            system.addForce(force)

        context = mm.Context(
            system,
            self._integrator,
            self._platform,
            self._platform_prop,
        )

        context.setPositions(
            openmmhelper.Positions.from_ase(np.array(atoms.positions))
        )
        context.setVelocities(
            openmmhelper.Velocities.from_ase(atoms.get_velocities())
        )

        self.context = context

    def get_atoms(self) -> Atoms:
        from ase.calculators.calculator import Calculator, all_changes

        class _Calculator(Calculator):
            implemented_properties = ['energy', 'forces']

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

        state = self.context.getState(
            getPositions=True,
            getVelocities=True,
            enforcePeriodicBox=False,
            getForces=True,
            getEnergy=True,
        )

        atoms = self._atoms.copy();

        atoms.set_positions(
            openmmhelper.Positions.from_openmm(
                state.getPositions(asNumpy=True)
            )
        )

        atoms.set_velocities(
            openmmhelper.Velocities.from_openmm(
                state.getVelocities(asNumpy=True)
            )
        )

        atoms.calc = _Calculator(
            energy=openmmhelper.Energy.from_openmm(
                energy=state.getPotentialEnergy()
            ),
            forces=openmmhelper.Forces.from_openmm(
                state.getForces(asNumpy=True)
            ),
        )

        return atoms

    def run(self, steps: int) -> None:
        steps_done = 0

        while steps_done < steps:
            to_run = min(
                map(
                    lambda x: x.next_match(steps_done),
                    self.callables
                )
            )

            self._integrator.step(to_run)
            steps_done += to_run

            atoms = self.get_atoms()

            for c in self.callables:
                if c.matches(steps_done):
                    c.call(steps_done, atoms)

    def close(self):
        del self._context
        del self._integrator
        gc.collect()
