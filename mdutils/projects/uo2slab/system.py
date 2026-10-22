import typing as tp
from abc import ABC, abstractmethod
from ase import Atoms
from ase.spacegroup import crystal
from ase.build import surface
from ase.constraints import FixAtoms
from ase.md.velocitydistribution import thermalize_momenta
from ase.geometry import find_mic
from ase.visualize import view
import matplotlib.pyplot as plt
from ase.visualize.plot import plot_atoms
import numpy as np
from collections import Counter
import enum
import random
import click
from tqdm import tqdm
from pathlib import Path
from ...system import System

class DefectType(enum.Enum):
    VACCANCY = enum.auto()
    ADATOM = enum.auto()

class Cluster(ABC):
    def __init__(self, *, defect_type: DefectType):
        self._init_pos: None | np.ndarray = None
        self._basis: None | tp.Tuple[np.ndarray, np.ndarray] = None
        self._defect_type = defect_type
        self._sequence: None | tp.Iterable[np.ndarray] = None

    def set_position(self, pos: np.ndarray) -> None:
        self._init_pos = pos

    def set_basis(self, basis: tp.Tuple[np.ndarray, np.ndarray]) -> None:
        self._basis = basis

    @property
    def defect_type(self) -> DefectType:
        return self._defect_type

    @abstractmethod
    def sequence(self) -> tp.Iterable[np.ndarray]:
        pass

    def add_uo2_molecule(self, position: np.ndarray, atoms: Atoms) -> None:
        a, b = self._basis
        c1 = (a - b / 2) / 3 * 2
        c2 = (b - a / 2) / 3 * 2
        delta = np.array([0.0, 0.0, 1.0])

        atoms += Atoms(
            symbols=["O", "U", "O"],
            positions = [
                (position + c1 + delta).tolist(),
                position.tolist(),
                (position + c2 - delta).tolist(),
            ]
        )

    def del_uo2_molecule(self, position: np.ndarray, atoms: Atoms) -> None:
        def delete_nearest_atom(r: np.ndarray, element: str, tol: float = 3.0) -> None:
            indices = np.array(
                [i for i, atom in enumerate(atoms)
                 if atom.symbol == element and
                 np.abs((atom.position - position))[2] < tol]
            )
            vectors = atoms.positions[indices] - r
            vectors_mic, distances = find_mic(
                vectors,
                cell=atoms.cell,
                pbc=atoms.pbc
            )

            nearest_idx = indices[np.argmin(distances)]
            del atoms[nearest_idx]

        delete_nearest_atom(position, "O")
        delete_nearest_atom(position, "O")
        delete_nearest_atom(position, "U")

    def modify(self, atoms: Atoms) -> None:
        if self._sequence is None:
            self._sequence = list(self.sequence())

        assert self._init_pos is not None

        for dp in self._sequence:
            position = self._init_pos + dp

            match self._defect_type:
                case DefectType.VACCANCY:
                    self.del_uo2_molecule(position, atoms);
                case DefectType.ADATOM:
                    self.add_uo2_molecule(position, atoms);


class Uo2SlabSystem(System):
    LATTICE_PARAMETER = 5.47
    SLAB_ORIENTATION = (1,1,1)
    N_LAYERS = 10
    REPEATITION = (10, 10, 1)
    RADIUS = 10.0
    VACUUM = 10.0

    def __init__(
        self,
        cluster: None | Cluster = None,
        temperature: None | float = None,
        lattice_parameter : float = LATTICE_PARAMETER,
        slab_orientation : tp.Tuple[int, int, int] = SLAB_ORIENTATION,
        n_layers : int = N_LAYERS,
        repeatition : tp.Tuple[int, int, int] = REPEATITION,
        vacuum: float = VACUUM,
    ) -> None:
        self._cluster = cluster
        self._lattice_parameter = lattice_parameter
        self._slab_orientation = slab_orientation
        self._n_layers = n_layers
        self._repeatition = repeatition
        self._vacuum = vacuum
        self._temperature = temperature

    def generate_bulk(self) -> Atoms:
        return crystal(
            symbols=["U", "O"],
            basis=[
                (0.0, 0.0, 0.0),
                (0.25, 0.25, 0.25)
            ],
            spacegroup=225,
            cellpar=[
                self._lattice_parameter,
                self._lattice_parameter,
                self._lattice_parameter,
                90, 90, 90
            ],
            primitive_cell=False,
        )

    def generate_slab(self, bulk : Atoms, tolerance=1e-3) -> Atoms:
        slab = surface(
            bulk,
            self._slab_orientation,
            self._n_layers,
            self._vacuum,
        ).repeat(self._repeatition)

        a, b = self.surface_basis(slab)
        c = (a - b / 2) / 3 * 2

        # get O indeces
        symbols = np.array(slab.get_chemical_symbols())
        o_idx = np.where(symbols == "O")
        o_atoms = slab.positions[o_idx]

        spacing = self._lattice_parameter / np.sqrt(3)

        top_z = np.max(o_atoms[:,2])
        bot_z = np.min(o_atoms[:,2])

        top_o_atoms = np.where(
            np.abs(slab.positions[:,2] - top_z) < tolerance
        )

        slab.positions[top_o_atoms] -= np.asarray(
            [0, 0, spacing * self._n_layers]
        )

        slab.pbc = True

        return slab

    def surface_basis(
        self,
        slab: Atoms
    ) -> tp.Tuple[np.ndarray, np.ndarray]:
        c = np.array(slab.cell)
        return (
            c[0] / self._repeatition[0] / 2,
            c[1] / self._repeatition[1] / 2
        )

    def add_clusters(
        self,
        slab: Atoms,
        tolerance: float = 1e-3
    ) -> None:
        if self._cluster is None:
            return

        a, b = self.surface_basis(slab)

        # get U indeces
        symbols = np.array(slab.get_chemical_symbols())
        u_idx = np.where(symbols == "U")

        u_positions = slab.positions[u_idx]

        spacing = self._lattice_parameter / np.sqrt(3)

        top_z = np.max(u_positions[:,2])
        bot_z = np.min(u_positions[:,2])

        reference_index = np.where(
            np.abs(top_z - u_positions[:,2]) < tolerance
        )[0][0]

        reference_position = u_positions[reference_index]
        opposite_position = reference_position - np.array(
            [0.0, 0.0, (self._n_layers - 1) * spacing]
        )

        match self._cluster.defect_type:
            case DefectType.VACCANCY:
                reference_position += np.array([0.0, 0.0, 1e-1])
                opposite_position -= np.array([0.0, 0.0, 1e-1])
            case DefectType.ADATOM:
                c = (a - b / 2) / 3 * 2
                reference_position += np.array([0.0, 0.0, spacing] + c)
                opposite_position += np.array([0.0, 0.0, -spacing] + c)

        self._cluster.set_basis(self.surface_basis(slab))
        self._cluster.set_position(reference_position)
        self._cluster.modify(slab)
        self._cluster.set_position(opposite_position)
        self._cluster.modify(slab)

    def freeze(self, atoms : Atoms) -> None:
        u_indices = [
            i for i, atom in enumerate(atoms)
            if atom.symbol == "U"
        ]

        atoms.set_constraint(FixAtoms(indices=u_indices))

        if self._temperature is not None:
            thermalize_momenta(atoms, self._temperature)

        vel = atoms.get_velocities()
        vel[u_indices] = 0.0
        atoms.set_velocities(vel)

    def create_system(self) -> Atoms:
        slab = self.generate_slab(
            self.generate_bulk()
        )

        self.add_clusters(slab)

        counts = Counter(slab.get_chemical_symbols())
        assert counts["U"] * 2 == counts["O"], "Total charge is not zero"

        self.freeze(slab)

        return slab

class SpiralCluster(Cluster):
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
        if k == 0:
            return [np.zeros((3,))]

        assert self._basis is not None
        a, b = self._basis
        vs = [a, b, b - a, -a, -b, -b + a]

        result = []

        for v in vs:
            result += [v] * k

        return result

    def sequence(self) -> tp.Iterable[np.ndarray]:
        assert self._basis is not None

        result = []

        r = np.zeros((3,))
        n_added = 0
        loop = 0

        while n_added < self.n_atoms:
            for v in self.basis_sequence(loop):
                if n_added >= self.n_atoms:
                    return result

                result.append(r.copy())

                r += v
                n_added += 1

            loop += 1
            r -= self._basis[1]

        return result


class RandomCluster(Cluster):
    def __init__(self, n_atoms: int, random_seed: int, *args, **kwargs):
        super().__init__(*args, **kwargs);
        self.n_atoms = n_atoms
        self.rng = random.Random()
        self.rng.seed(random_seed)

    def sequence(self) -> tp.Iterable[np.ndarray]:
        if self.n_atoms == 0:
            return

        assert self._basis is not None

        a, b = self._basis
        basis_seq = [a, b, b - a, -a, -b, -b + a]


        r = np.zeros((3,))

        sequence = [r.copy()]

        def has_in_seq(r: np.ndarray, tol=1e-3) -> bool:
            for s in sequence:
                if np.all(np.isclose(s, r, atol=tol)):
                    return True
            return False

        while len(sequence) < self.n_atoms:
            while has_in_seq(r):
                r += self.rng.choice(basis_seq)

            sequence.append(r.copy())

        return sequence

class ClusterType(enum.Enum):
    SPIRAL = enum.auto()
    RANDOM = enum.auto()

@click.group()
def cli(): pass

@cli.command("view")
@click.option(
    "--dtype", "defect_type", type=click.Choice(DefectType),
    help="Defect type",
)
@click.option(
    "--ctype", "cluster_type", type=click.Choice(ClusterType),
    help="Cluster type",
)
@click.option(
    "--random-seed", "random_seed", type=click.INT, default=4,
    help="Random seed"
)
@click.option(
    "--ndef", "n_defects", type=click.INT, default=0,
    help="Number of defects"
)
@click.option(
    "--nlayers", "n_layers", type=click.INT, default=12,
    help="Number of datomic layers"
)
@click.option(
    "--rep", "replicate", type=click.INT, default=5,
    help="Replication number"
)
def view_system(
    defect_type: DefectType,
    cluster_type: ClusterType,
    random_seed: int,
    n_defects: int,
    n_layers: int,
    replicate: int,
):
    cluster = None

    match cluster_type:
        case ClusterType.SPIRAL:
            cluster = SpiralCluster(n_defects, defect_type=defect_type)
        case ClusterType.RANDOM:
            cluster = RandomCluster(n_defects, random_seed, defect_type=defect_type)

    system = Uo2SlabSystem(
        cluster,
        vacuum=160.0,
        repeatition=(replicate,replicate,1),
        n_layers=n_layers,
    )
    view(system.create_system())


@cli.command("render-clusters")
@click.option(
    "--random-seed", "random_seed", type=click.INT, default=4,
    help="Random seed"
)
@click.option(
    "--ndef", "n_defects", type=click.INT, default=0,
    help="Number of defects"
)
@click.option(
    "--rep", "replicate", type=click.INT, default=5,
    help="Replication number"
)
@click.option(
    "--n", "n_systems", type=click.INT,
    help="Number of systems"
)
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
def render_clusters(
    random_seed: int,
    n_defects: int,
    replicate: int,
    n_systems: int,
    working_directory: Path,
):
    rng = random.Random()
    rng.seed(random_seed)

    for i in tqdm(range(n_systems)):
        plt.cla()
        seed = rng.getrandbits(16)
        atoms = Uo2SlabSystem(
            RandomCluster(n_defects, seed, defect_type=DefectType.VACCANCY),
            vacuum=160.0,
            repeatition=(replicate,replicate,1),
            n_layers=12,
        ).create_system()

        symbols = np.array(atoms.get_chemical_symbols())
        u_idx = np.where(symbols == "U")
        u_positions = atoms.positions[u_idx]
        top_z = np.max(u_positions[:,2])
        to_del = [i for i, atom in enumerate(atoms)
                  if np.abs(atom.position[2] - top_z) > 3.0 or atom.symbol != "U"]
        del atoms[to_del]

        plot_atoms(atoms)
        plt.title(f"N = {n_defects}, seed = 0x{seed:04x}")
        plt.savefig(working_directory / f"{i}.png")
        plt.close()


if __name__ == "__main__":
    cli()


