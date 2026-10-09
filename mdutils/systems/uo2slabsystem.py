import typing as tp
from abc import ABC, abstractmethod
from ase import Atoms
from ase.spacegroup import crystal
from ase.build import surface
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
import numpy as np
from collections import Counter
from ..system import System

class Uo2SlabSystem(ABC):
    LATTICE_PARAMETER = 5.47
    SLAB_ORIENTATION = (1,1,1)
    N_LAYERS = 10
    REPEATITION = (10, 10, 1)
    RADIUS = 10.0
    VACUUM = 10.0

    def __init__(
        self,
        N,
        temperature: float,
        lattice_parameter : float = LATTICE_PARAMETER,
        slab_orientation : tp.Tuple[int, int, int] = SLAB_ORIENTATION,
        n_layers : int = N_LAYERS,
        repeatition : tp.Tuple[int, int, int] = REPEATITION,
        radius : float = RADIUS,
        vacuum: float = VACUUM,
    ) -> None:
        assert n_layers % 2 == 0, "Number of layers should be even"

        self.lattice_parameter = lattice_parameter
        self.slab_orientation = slab_orientation
        self.n_layers = n_layers
        self.repeatition = repeatition
        self.radius = radius
        self.vacuum = vacuum
        self.N = N
        self.temperature = temperature

    def generate_bulk(self) -> Atoms:
        return crystal(
            symbols=["U", "O"],
            basis=[
                (0.0, 0.0, 0.0),
                (0.25, 0.25, 0.25)
            ],
            spacegroup=225,
            cellpar=[
                self.lattice_parameter,
                self.lattice_parameter,
                self.lattice_parameter,
                90, 90, 90
            ],
            primitive_cell=False,
        )

    def uo2_molecule(
        self,
        slab: Atoms,
        x: float,
        y : float,
        z : float
    ) -> Atoms:
        a, b = self.surface_basis(slab)
        c1 = (a - b / 2) / 3 * 2
        c2 = (b - a / 2) / 3 * 2
        delta = np.array([0.0, 0.0, 1.0])

        ref = np.array([x, y, z])

        return Atoms(
            symbols=["O", "U", "O"],
            positions = [
                (ref + c1 + delta).tolist(),
                ref.tolist(),
                (ref + c2 - delta).tolist(),
            ]
        )

    def generate_slab(self, bulk : Atoms, tolerance=1e-3) -> Atoms:
        slab = surface(
            bulk,
            self.slab_orientation,
            self.n_layers,
            self.vacuum,
        ).repeat(self.repeatition)

        a, b = self.surface_basis(slab)
        c = (a - b / 2) / 3 * 2

        # get O indeces
        symbols = np.array(slab.get_chemical_symbols())
        o_idx = np.where(symbols == "O")
        o_atoms = slab.positions[o_idx]

        spacing = self.lattice_parameter / np.sqrt(3)

        top_z = np.max(o_atoms[:,2])
        bot_z = np.min(o_atoms[:,2])

        top_o_atoms = np.where(
            np.abs(slab.positions[:,2] - top_z) < tolerance
        )

        slab.positions[top_o_atoms] -= np.asarray(
            [0, 0, spacing * self.n_layers]
        ) + c

        slab.pbc = True

        return slab

    def surface_basis(
        self,
        slab: Atoms
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        c = np.array(slab.cell)
        return (
            c[0] / self.repeatition[0] / 2,
            c[1] / self.repeatition[1] / 2
        )

    def form_cluster(
        self,
        slab: Atoms,
        n_atoms: int,
        reference: np.ndarray,
        basis: tp.Tuple[np.ndarray, np.ndarray]
    ) -> Atoms:
        def basis_sequence(k : int):
            a, b = basis
            vs = [a, b, b - a, -a, -b, -b + a]

            result = []

            for v in vs:
                result += [v] * k

            return result

        assert n_atoms > 0

        n_added = 1
        loop = 1
        x, y, z = reference

        cluster = self.uo2_molecule(slab, x, y, z)

        while True:
            if n_added >= n_atoms:
                break

            x, y, z = np.array([x, y, z]) - basis[1]

            for v in basis_sequence(loop):
                if n_added >= n_atoms:
                    return cluster

                x, y, z = np.array([x, y, z]) + v

                cluster += self.uo2_molecule(slab, x, y, z)

                n_added += 1

            loop += 1

        return cluster

    def add_clusters(
        self,
        slab: Atoms,
        n_atoms: int,
        tolerance: float = 1e-3
    ) -> Atoms:
        a, b = self.surface_basis(slab)

        # get U indeces
        symbols = np.array(slab.get_chemical_symbols())
        u_idx = np.where(symbols == "U")

        u_positions = slab.positions[u_idx]

        spacing = self.lattice_parameter / np.sqrt(3)

        top_z = np.max(u_positions[:,2])
        bot_z = np.min(u_positions[:,2])

        reference_index = np.where(
            np.abs(top_z - u_positions[:,2]) < tolerance
        )[0][0]

        reference_position = u_positions[reference_index]
        opposite_position = reference_position - np.array(
            [0.0, 0.0, (self.n_layers - 1) * spacing]
        )

        c = (a - b / 2) / 3 * 2
        reference_position += np.array([0.0, 0.0, spacing] + c)
        opposite_position += np.array([0.0, 0.0, -spacing] + c)

        basis = self.surface_basis(slab)

        return (
            slab
            + self.form_cluster(
                slab,
                n_atoms,
                reference_position,
                basis
            )
            + self.form_cluster(
                slab,
                n_atoms,
                opposite_position,
                basis
            )
        )

    def freeze(self, atoms : Atoms) -> None:
        u_indices = [
            i for i, atom in enumerate(atoms)
            if atom.symbol == "U"
        ]

        atoms.set_constraint(FixAtoms(indices=u_indices))

        MaxwellBoltzmannDistribution(
            atoms,
            temperature_K=self.temperature,
        )

        vel = atoms.get_velocities()
        vel[u_indices] = 0.0
        atoms.set_velocities(vel)

    def create_system(self) -> Atoms:
        slab = self.generate_slab(
            self.generate_bulk()
        )

        slab = self.add_clusters(slab, self.N)

        counts = Counter(slab.get_chemical_symbols())
        assert counts["U"] * 2 == counts["O"], "Total charge is not zero"

        self.freeze(slab)

        return slab
