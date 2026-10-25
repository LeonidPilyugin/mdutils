from __future__ import annotations

import typing as tp
from pathlib import Path

import click
import numpy as np
import openmm as mm
import pandas as pd
from ase import Atoms
from ase.constraints import FixAtoms
from ase.io import read
from ase.neighborlist import neighbor_list

from ...openmmcompat import OpenMmSolver as Solver
from ...utils.parallel import Parallel, Worker
from ...utils.platform import Platform
from ...utils.timer import Timer
from .forces import load_forces


def get_cut_horizontal_u_bonds_number(
    slab: Atoms,
    cutoff: float = 4.2,
    tol: float = 1e-1,
) -> int:
    symbols = np.array(slab.get_chemical_symbols())
    u_indices = np.where(symbols == "U")[0]

    cutoff = 4.2  # Angstrom

    # Surface normal; assumes cell[0] and cell[1] span the (111) plane
    cell = slab.cell.array
    normal = np.cross(cell[0], cell[1])
    normal /= np.linalg.norm(normal)

    # Directed neighbor pairs and bond vectors
    i, j, D = neighbor_list("ijD", slab, cutoff)

    # U-U bonds only
    mask = (symbols[i] == "U") & (symbols[j] == "U")
    i, j, D = i[mask], j[mask], D[mask]

    # Bonds parallel to the (111) plane
    parallel = np.abs(D @ normal) < tol

    # Count in-plane U neighbors per uranium atom
    coord_parallel = np.bincount(
        i[parallel], minlength=len(slab)
    )

    # Six nearest in-plane U neighbors in ideal bulk fluorite UO2
    expected = 6
    missing_per_U = np.maximum(
        0, expected - coord_parallel[u_indices]
    )

    return missing_per_U.sum()


def perform_file(
    path: Path,
    platform: Platform,
    gpu_idx: int,
    freeze_u: bool = True,
) -> tuple[float, int]:
    atoms = read(
        path,
        format="lammps-data",
        Z_of_type={1: 8, 2: 92},  # O: type 1, U: type 2
    )

    assert isinstance(atoms, Atoms)

    charges = np.zeros(len(atoms))
    for i, atom in enumerate(atoms):
        if atom.symbol == "O":
            charges[i] = -1.37246
        elif atom.symbol == "U":
            charges[i] = 2.74492
        else:
            raise ValueError(f"Unexpected element: {atom.symbol}")

    atoms.set_initial_charges(charges)

    missing_u_bonds = get_cut_horizontal_u_bonds_number(atoms)

    if freeze_u:
        u_indices = [
            atom.index
            for atom in atoms
            if atom.symbol == "U"
        ]

        atoms.set_constraint(FixAtoms(indices=u_indices))

    with Solver(
        mm.VerletIntegrator(1e-3),
        mm.Platform.getPlatformByName(Platform.as_str(platform)),
        { "Precision": "mixed", "DeviceIndex": str(gpu_idx) },
        *load_forces()
    ) as solver:
        solver.load_system(atoms)

        eva2mm = 96.4853

        solver.minimize(
            tolerance=0.1 * eva2mm,
            maxIterations=10000,
        )

        result = (
            solver.get_atoms().get_potential_energy(),
            missing_u_bonds,
        )

    return result;

def filegenerator(
    path: Path,
    every: int,
    suffix: str = ".lammpsdata"
) -> tp.Iterator[tuple[Path, Path]]:
    for p in filter(lambda x: x.is_dir(), path.iterdir()):
        for f in sorted(
                filter(
                    lambda x: x.suffix == suffix,
                    (p / "nomean").iterdir()
                ),
                key=lambda x: int(x.stem)
            )[::every]:
            yield f, p

class MinimizeWorker(
    Worker[
        [tuple[Path, Path], int],
        tuple[int, float, float, int, int | None],
    ]
):
    def __init__(
        self,
        platform: Platform,
        freeze_u: bool,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._platform = platform
        self._freeze_u = freeze_u

    def __call__(
        self,
        tup: tuple[Path, Path],
        gpu_idx: int,
    ) -> tuple[int, float, float, int, None | int]:
        f, p = tup
        spl = p.name.split("-")
        data = ()
        if len(spl) == 2:
            data = float(spl[0]), int(spl[1]), None
        else:
            assert len(spl) == 3
            data = float(spl[0]), int(spl[1]), int(spl[2], 16)


        e, b = perform_file(
            f,
            self._platform,
            gpu_idx,
            freeze_u=self._freeze_u
        )
        self.update()

        return b, e, data[0], data[1], data[2]

@click.command()
@click.option(
    "--platform", "platform", default=Platform.HIP, type=click.Choice(Platform),
    help="Computational platform",
)
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
@click.option(
    "--every", "every", type=click.INT,
    help="Working directory"
)
@click.option(
    "--filename", "filename", type=click.STRING, default="minimize.csv",
    help="Working directory"
)
@click.option(
    "--freeze", "freeze_u", is_flag=True, default=False,
    help="Freeze U atoms during energy minimization"
)
@click.option(
    "--parallel", "parallel_workers", default=4, type=click.INT,
    help="Number of parallel workers",
)
@click.option(
    "--gpu", "gpus", type=click.INT, multiple=True,
    help="Available GPU indices",
)
def main(
    platform: Platform,
    working_directory: Path,
    every: int,
    filename: str,
    freeze_u: bool,
    parallel_workers: int,
    gpus: tp.Sequence[int],
) -> None:
    total = len(list(filegenerator(working_directory, every)))
    result = None

    with Parallel(parallel=parallel_workers, delay=1.0) as parallel:
        result = parallel.map(
            MinimizeWorker(
                platform,
                freeze_u,
                guard=parallel.guard,
            ),
            filegenerator(working_directory, every),
            [gpus[i % len(gpus)] for i in range(total)],
        )

    assert result is not None

    pd.DataFrame({
        "e": [r[1] for r in result],
        "dist": [r[2] for r in result],
        "n": [r[3] for r in result],
        "b": [r[0] for r in result],
        "seed": [r[1] for r in result],
    }).to_csv(working_directory / filename)


if __name__ == "__main__":
    with Timer():
        main()
