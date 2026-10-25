import typing as tp

import click
import numpy as np
import openmm as mm
from ase import Atoms

from ...openmmcompat import OpenMmSolver as Solver
from ...utils.platform import Platform
from ...utils.timer import Timer
from .forces import load_forces
from .system import Uo2SlabSystem as System


def create_system(
    n_layers: int,
    vacuum: float,
    repeat: int
) -> Atoms:
    atoms = System(
        vacuum=vacuum,
        repeatition=(repeat,repeat,1),
        n_layers=n_layers + 3 - n_layers % 3
    ).create_system()

    for _ in range(3 - n_layers % 3):
        u_indices = [
                int(atom.index) # pyright: ignore[reportArgumentType]
            for atom in atoms
            if atom.symbol == "U"
        ]
        u_pos = atoms.positions[u_indices] # pyright: ignore[reportCallIssue]
        top_u = np.max(u_pos[:,2])
        to_del = [i for i, a in enumerate(atoms)
                  if a.symbol == "U" and
                  abs(atoms.positions[i,2] - top_u) < 1e-3]

        del atoms[to_del]

        for _ in range(2):
            o_indices = [
                int(atom.index) # pyright: ignore[reportArgumentType]
                for atom in atoms
                if atom.symbol == "O"
            ]
            o_pos = atoms.positions[o_indices] # pyright: ignore[reportCallIssue]
            top_o = np.max(o_pos[:,2])
            to_del = [i for i, a in enumerate(atoms)
                      if a.symbol == "O"
                      and abs(atoms.positions[i,2] - top_o) < 1e-3]
            del atoms[to_del]

    return atoms

@click.command()
@click.option(
    "--platform", default=Platform.HIP, type=click.Choice(Platform),
    help="Computational platform",
)
@click.option(
    "--repeat", default=7, type=click.INT,
    help="Repeat number",
)
@click.option(
    "--layers", type=click.INT, multiple=True,
    help="Number of atom layers",
)
@click.option(
    "--vacuum", default=100.0, type=click.FLOAT,
    help="Vacuum width",
)
def main(
    platform: Platform,
    repeat: int,
    layers: tp.Sequence[int],
    vacuum: float,
) -> None:
    assert len(layers) >= 2, f"Layers length is {len(layers)} < 2"

    es = []
    ns = []

    for l in layers:
        system = create_system(l, vacuum, repeat)
        with Solver(
            mm.VerletIntegrator(1e-3),
            mm.Platform.getPlatformByName(Platform.as_str(platform)),
            { "Precision": "mixed", "DeviceIndex": str(0) },
            *load_forces()
        ) as solver:
            solver.load_system(system)
            es.append(solver.get_atoms().get_potential_energy())
            ns.append(len(system))

    a, _ = np.polyfit(ns, es, 1)
    print(f"Cogesion energy is {a * 6:.3f} eV / 2x UO2 molecule")

if __name__ == "__main__":
    with Timer():
        main()
