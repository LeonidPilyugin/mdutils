import typing as tp
from dataclasses import dataclass
import openmm as mm
import numpy as np
from ase import Atoms
from ase.constraints import FixAtoms
from ase.units import fs
from ..solver import Solver

class OpenMmSolver(Solver):
    @dataclass
    class Units:
        length: mm.unit.Unit
        time: mm.unit.Unit
        energy: mm.unit.Unit
        mass: mm.unit.Unit

        @property
        def velocity(self) -> mm.unit.Unit:
            return self.length / self.time

        @property
        def force(self) -> mm.unit.Unit:
            return self.energy / self.length

    def __init__(self) -> None:
        super().__init__()

        self.units = OpenMmSolver.Units(
            length = mm.unit.angstrom,
            time = mm.unit.femtoseconds,
            energy = mm.unit.kilojoule_per_mole,
            mass = mm.unit.atom_mass_units,
        )

    def load_from_dict(self, dct: tp.Dict[str, tp.Any]) -> None:
        integrator_class = getattr(mm, dct["integrator"]["type"])
        self.integrator = integrator_class(*dct["integrator"]["arguments"])
        # self.integrator = mm.VerletIntegrator(
        #     0.25 * mm.unit.femtoseconds
        # )

        self.platform = dct["platform"]

        self.potentials = []
        self.potential_data = []

        for potential_data in dct["potentials"]:
            with open(potential_data["path"], "r") as f:
                self.potentials.append(
                    mm.XmlSerializer.deserialize(
                        f.read()
                    )
                )

                self.potential_data.append(
                    potential_data.get(
                        "particles", None
                    )
                )

    def load_system(self, atoms: Atoms) -> None:
        self._atoms = atoms.copy()

        cell = np.array(atoms.cell) * self.units.length
        positions = np.array(atoms.positions) * self.units.length
        velocities = (
            np.array(atoms.get_velocities()) * fs
        ) * self.units.velocity
        masses = np.array(atoms.get_masses()) * self.units.mass
        types = atoms.get_chemical_symbols()

        system = mm.System()
        system.setDefaultPeriodicBoxVectors(*cell)

        for constraint in atoms.constraints:
            if isinstance(constraint, FixAtoms):
                masses[constraint.get_indices()] = 0.0 * self.units.mass

        for mass in masses:
            system.addParticle(mass)

        for i in range(len(self.potentials)):
            force = self.potentials[i]
            data = self.potential_data[i]

            if hasattr(force, "addParticle"):
                for i, _ in enumerate(masses):
                    force.addParticle(*data[types[i]])

            system.addForce(force)

        platform = mm.Platform.getPlatformByName(
            self.platform["type"]
        )

        context = mm.Context(
            system,
            self.integrator,
            platform,
            self.platform["arguments"]
        )

        context.setPositions(positions)
        context.setVelocities(velocities)

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
            state.getPositions(asNumpy=True)
                .value_in_unit(self.units.length)
        )

        atoms.set_velocities(
            state.getVelocities(asNumpy=True)
                .value_in_unit(self.units.velocity) / fs
        )

        atoms.calc = _Calculator(
            energy=state.getPotentialEnergy()
                .value_in_unit(
                    mm.unit.kilojoule_per_mole
                ) / 96.4853321233,
            forces=state.getForces(asNumpy=True)
                .value_in_unit(
                    mm.unit.kilojoule_per_mole / mm.unit.angstrom
                ) / 96.4853321233
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

            self.integrator.step(to_run)
            steps_done += to_run

            atoms = self.get_atoms()

            for c in self.callables:
                if c.matches(steps_done):
                    c.call(steps_done, atoms)
