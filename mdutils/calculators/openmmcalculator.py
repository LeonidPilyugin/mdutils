import numpy as np
import openmm as mm
from openmm import unit
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
        dct: tp.Dict[str, tp.Any],
        **kwargs,
    ):
        super().__init__(**kwargs)

        integrator_class = getattr(mm, dct["integrator"]["type"])
        self.integrator = integrator_class(*dct["integrator"]["arguments"])

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

        cell = openmmhelper.Cell.from_ase(atoms.cell.array)
        positions = openmmhelper.Positions.from_ase(atoms.get_positions())
        types = atoms.get_chemical_symbols()

        system = mm.System()
        system.setDefaultPeriodicBoxVectors(*cell)

        for _ in positions:
            system.addParticle(0)

        for i in range(len(self.potentials)):
            force = self.potentials[i]
            data = self.potential_data[i]

            if hasattr(force, "addParticle"):
                for i, _ in enumerate(positions):
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

        self.context = context

    def calculate(
        self,
        atoms=None,
        properties=("energy", "forces"),
        system_changes=all_changes,
    ):
        super().calculate(atoms, properties, system_changes)

        self.context.setPositions(
            openmmhelper.Positions.from_ase(atoms.get_positions())
        )

        self.context.setPeriodicBoxVectors(
            *openmmhelper.Cell.from_ase(np.array(atoms.cell))
        )

        state = self.context.getState(
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
        del self.context
        del self.integrator

