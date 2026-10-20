import typing as tp
from abc import ABC, abstractmethod
from ase import Atoms
from ase.spacegroup import crystal
from ase.build import surface
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.geometry import find_mic
import numpy as np
from collections import Counter
import enum
from ...system import System

class DefectType(enum.Enum):
    VACCANCY = enum.auto()
    ADATOM = enum.auto()

class Uo2SlabSystem(System):
    class Cluster(ABC):
        def __init__(self):
            self.position = np.zeros((3,))
            self.basis = (np.zeros((3,)), np.zeros((3,)))

        def uo2_molecule(self, position: np.ndarray) -> Atoms:
            a, b = self.basis
            c1 = (a - b / 2) / 3 * 2
            c2 = (b - a / 2) / 3 * 2
            delta = np.array([0.0, 0.0, 1.0])

            return Atoms(
                symbols=["O", "U", "O"],
                positions = [
                    (position + c1 + delta).tolist(),
                    position.tolist(),
                    (position + c2 - delta).tolist(),
                ]
            )

        def set_position(self, pos: np.ndarray) -> None:
            self.position = pos

        def set_basis(self, basis: tp.Tuple[np.ndarray, np.ndarray]) -> None:
            self.basis = basis

        @abstractmethod
        def delete_atoms(self, atoms: Atoms) -> None:
            pass

        @abstractmethod
        def form_cluster(self) -> Atoms:
            pass


    LATTICE_PARAMETER = 5.47
    SLAB_ORIENTATION = (1,1,1)
    N_LAYERS = 10
    REPEATITION = (10, 10, 1)
    RADIUS = 10.0
    VACUUM = 10.0

    def __init__(
        self,
        cluster: Cluster,
        temperature: float = 3000.0,
        lattice_parameter : float = LATTICE_PARAMETER,
        slab_orientation : tp.Tuple[int, int, int] = SLAB_ORIENTATION,
        n_layers : int = N_LAYERS,
        repeatition : tp.Tuple[int, int, int] = REPEATITION,
        vacuum: float = VACUUM,
        defect_type: DefectType = DefectType.VACCANCY,
    ) -> None:
        self.cluster = cluster
        self.lattice_parameter = lattice_parameter
        self.slab_orientation = slab_orientation
        self.n_layers = n_layers
        self.repeatition = repeatition
        self.vacuum = vacuum
        self.temperature = temperature
        self.delete = defect_type is DefectType.VACCANCY

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
        )

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

    def add_clusters(
        self,
        slab: Atoms,
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

        if self.delete:
            reference_position += np.array([0.0, 0.0, 1e-1])
            opposite_position -= np.array([0.0, 0.0, 1e-1])
        else:
            c = (a - b / 2) / 3 * 2
            reference_position += np.array([0.0, 0.0, spacing] + c)
            opposite_position += np.array([0.0, 0.0, -spacing] + c)

        self.cluster.set_basis(self.surface_basis(slab))

        if self.delete:
            self.cluster.set_position(reference_position)
            self.cluster.delete_atoms(slab)
            self.cluster.set_position(opposite_position)
            self.cluster.delete_atoms(slab)
            return slab

        self.cluster.set_position(reference_position)
        top_cluster = self.cluster.form_cluster()
        self.cluster.set_position(opposite_position)
        bot_cluster = self.cluster.form_cluster()

        return slab + top_cluster + bot_cluster

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

        slab = self.add_clusters(slab)

        counts = Counter(slab.get_chemical_symbols())
        assert counts["U"] * 2 == counts["O"], "Total charge is not zero"

        self.freeze(slab)

        return slab

class SpiralCluster(Uo2SlabSystem.Cluster):
    def __init__(
        self,
        n_atoms: int,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        assert n_atoms >= 0

        self.n_atoms = n_atoms

    def basis_sequence(self, k: int):
        a, b = self.basis
        vs = [a, b, b - a, -a, -b, -b + a]

        result = []

        for v in vs:
            result += [v] * k

        return result

    def form_cluster(self) -> Atoms:
        if self.n_atoms == 0:
            return Atoms()

        n_added = 1
        loop = 1
        r = self.position.copy()

        cluster = self.uo2_molecule(r)

        while n_added < self.n_atoms:
            r -= self.basis[1]

            for v in self.basis_sequence(loop):
                if n_added >= self.n_atoms:
                    return cluster

                r += v
                n_added += 1

                cluster += self.uo2_molecule(r)

            loop += 1

        return cluster

    def delete_uo2_molecule(self, r: np.ndarray, atoms: Atoms) -> None:
        def delete_nearest_atom(r: np.ndarray, element: str) -> None:
            indices = np.array([i for i, atom in enumerate(atoms)
                                if atom.symbol == element])
            vectors = atoms.positions[indices] - r
            vectors_mic, distances = find_mic(
                vectors,
                cell=atoms.cell,
                pbc=atoms.pbc
            )

            nearest_idx = indices[np.argmin(distances)]
            del atoms[nearest_idx]

        delete_nearest_atom(r, "O")
        delete_nearest_atom(r, "O")
        delete_nearest_atom(r, "U")


    def delete_atoms(self, atoms: Atoms) -> None:
        if self.n_atoms == 0:
            return

        n_added = 1
        loop = 1
        r = self.position.copy()
        self.delete_uo2_molecule(r, atoms)

        while n_added < self.n_atoms:
            r -= self.basis[1]

            for v in self.basis_sequence(loop):
                if n_added >= self.n_atoms:
                    return

                r += v
                n_added += 1

                self.delete_uo2_molecule(r, atoms)

            loop += 1


if __name__ == "__main__":
    from ase.visualize import view

    system = Uo2SlabSystem(
        SpiralCluster(7),
        3000.0,
        vacuum=160.0,
        repeatition=(5,5,1),
        n_layers=12,
        defect_type=DefectType.VACCANCY,
    )

    view(system.create_system(), viewer="paraview")
