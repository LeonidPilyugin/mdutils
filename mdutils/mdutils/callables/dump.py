from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from ..solver import Solver

class Dump(Solver.Callable):
    def __init__(
        self,
        path: Path,
        every: int,
        mean: int,
        fstring: str = "{step:010d}.lammpsdata",
        format_: str = "lammps-data",
        log: str = "log",
    ) -> None:
        if not path.exists():
            path.mkdir()

        assert every >= mean
        assert mean > 0

        self.path = path
        self.every = every
        self.mean = mean
        self.fstring = fstring
        self.format = format_
        self.log = log

        with open(path / log, "w") as f:
            f.write("step\tk_e\tp_e\tt_e\ttemperature\n")

        self.array = []

    def matches(self, step: int) -> bool:
        return step % self.every >= self.every - self.mean

    def next_match(self, step: int) -> int:
        if step % self.every >= self.every - self.mean:
            return 1

        return self.every - self.mean - step % self.every

    def call(self, step: int, atoms: Atoms) -> None:
        if len(self.array) < self.mean - 1:
            a = atoms.copy()
            a.calc = atoms.calc
            self.array.append(a)
            return

        atoms_ = atoms.copy()
        atoms_.calc = atoms.calc

        k_e = atoms_.get_kinetic_energy()
        p_e = atoms_.get_potential_energy()
        t_e = atoms_.get_total_energy()
        temperature = atoms_.get_temperature()

        if len(self.array) > 1:
            atoms_.set_positions(
                np.array(
                    [a.get_positions() for a in self.array]
                ).mean(axis=0)
            )

            k_e = np.mean([a.get_kinetic_energy() for a in self.array])
            p_e = np.mean([a.get_potential_energy() for a in self.array])
            t_e = np.mean([a.get_total_energy() for a in self.array])
            temperature = np.mean([a.get_temperature() for a in self.array])

        self.array.clear()

        write(
            self.path / self.fstring.format(step=step),
            atoms_,
            format=self.format,
        )

        with open(self.path / self.log, "a") as f:
            f.write(f"{step}\t{k_e}\t{p_e}\t{t_e}\t{temperature}\n")
