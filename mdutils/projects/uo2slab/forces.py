from __future__ import annotations

import typing as tp
from pathlib import Path

import openmm as mm


def load_forces(
    root: Path | None = None
) -> tuple[tp.Iterable[mm.Force], tp.Iterable[dict[str, tp.Any]]]:
    if root is None:
        root = Path("~/potentials").expanduser()

    forces = []
    for i in range(3):
        with open(root / f"uo2_xe{i}.xml") as f:
            forces.append(mm.XmlSerializer.deserialize(f.read()))

    forcedata = [{
            "O": [[1, 0, 0]],
            "U": [[0, 1, 0]],
            "Xe": [[0, 0, 1]],
        }, {
            "O": [-1.37246, 0.1, 0.0],
            "U": [2.74492, 0.1, 0.0],
            "Xe": [0.0, 0.1, 0.0],
        }, {}
    ]

    return forces, forcedata

