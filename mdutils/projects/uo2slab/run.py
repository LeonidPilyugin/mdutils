from .system import Uo2SlabSystem as System
from .system import Cluster, SpiralCluster, RandomCluster
from .system import DefectType
from ...solvers import OpenMmSolver as Solver
from ...utils.parallel import Parallel, Worker, ProgressCallable
from ...utils.platform import Platform
from ... import callables as mdc
from .forces import load_forces
from pathlib import Path
import openmm as mm
import sys
import shutil
from itertools import product
import multiprocessing as mp
import typing as tp
from ase import Atoms
import click
from dataclasses import dataclass
import random

@dataclass
class Params:
    gpu_idx: int
    vacuum: float
    temperature: float
    repeatition: tp.Tuple[int, int, int]
    n_layers: int
    cluster: Cluster
    subdir: str

class RunWorker(Worker):
    def __init__(
        self,
        platform: Platform,
        run_steps: int,
        mean_every: int | None,
        submit_every: int,
        working_path: Path,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._platform = platform
        self._run_steps = run_steps
        self._mean_every = mean_every
        self._submit_every = submit_every
        self._working_path = working_path
        self._reporter = ProgressCallable(self._guard, submit_every)

    def __call__(self, params: Params, *args) -> tp.Any:
        path = self._working_path / params.subdir
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True)

        system = System(
            params.cluster,
            temperature=params.temperature,
            vacuum=params.vacuum,
            repeatition=params.repeatition,
            n_layers=params.n_layers,
        ).create_system()

        with Solver(
            mm.LangevinIntegrator(params.temperature, 1e-2, 1e-3),
            mm.Platform.getPlatformByName(Platform.as_str(self._platform)),
            { "Precision": "mixed", "DeviceIndex": str(params.gpu_idx) },
            *load_forces()
        ) as solver:
            solver.load_system(system)

            if self._mean_every is not None:
                solver.attach(
                    mdc.Dump(
                        path / "mean",
                        self._submit_every,
                        self._mean_every,
                    )
                )

            solver.attach(
                mdc.Dump(
                    path / "nomean",
                    self._submit_every,
                    1,
                )
            )

            solver.attach(self._reporter)

            solver.run(self._run_steps)

    def total(self) -> int:
        return self._run_steps

@click.group()
def cli() -> None:
    pass

@cli.command("run-spiral")
@click.option(
    "--temperature", "temperature", default=3000.0, type=click.FLOAT,
    help="Temperature of system in K",
)
@click.option(
    "--platform", "platform", default=Platform.HIP,
    type=click.Choice(Platform), help="Computational platform",
)
@click.option(
    "--vacuum", "vacuum_list", default=(160.0,), type=click.FLOAT,
    multiple=True, help="Vacuum width in angstroms",
)
@click.option(
    "--defect", "defect_list", default=(0,), type=click.INT,
    multiple=True, help="Defect count in a single cluster",
)
@click.option(
    "--run", "run_steps", default=1000000, type=click.INT,
    help="Number of steps to run",
)
@click.option(
    "--mean", "mean_every", default=1000, type=click.INT,
    help="Number of steps in mean frame",
)
@click.option(
    "--submit", "submit_every", default=10000, type=click.INT,
    help="Number of submitting interval in steps",
)
@click.option(
    "--parallel", "parallel_workers", default=4, type=click.INT,
    help="Number of parallel workers",
)
@click.option(
    "--gpu", "gpus", type=click.INT, multiple=True,
    help="Available GPU indices",
)
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
@click.option(
    "--layers", "n_layers", default=12, type=click.INT,
    help="Number of atom layers",
)
@click.option(
    "--repeat", "repeat", default=7, type=click.INT,
    help="Number of horizontal repliation",
)
def run_spiral(
    temperature: float,
    platform: Platform,
    vacuum_list: tp.Sequence[float],
    defect_list: tp.Sequence[int],
    run_steps: int,
    mean_every: int,
    submit_every: int,
    parallel_workers: int,
    gpus: tp.Sequence[int],
    working_directory: Path,
    n_layers: int,
    repeat: int,
) -> None:
    def params_generator() -> tp.Iterable[Params]:
        for i, (v, d) in enumerate(product(vacuum_list, defect_list)):
            yield Params(
                gpu_idx=gpus[i % len(gpus)],
                vacuum=v,
                temperature=temperature,
                repeatition=(repeat,repeat,1),
                n_layers=n_layers,
                cluster=SpiralCluster(d, defect_type=DefectType.VACCANCY),
                subdir=f"{v}-{d}",
            )

    with Parallel(parallel=parallel_workers, delay=1.0) as parallel:
        parallel.map(
            RunWorker(
                platform=platform,
                run_steps=run_steps,
                mean_every=mean_every,
                submit_every=submit_every,
                working_path=working_directory,
                guard=parallel.guard,
            ),
            params_generator()
        )


@cli.command("run-random")
@click.option(
    "--temperature", "temperature", default=3000.0, type=click.FLOAT,
    help="Temperature of system in K",
)
@click.option(
    "--platform", "platform", default=Platform.HIP,
    type=click.Choice(Platform), help="Computational platform",
)
@click.option(
    "--vacuum", "vacuum_list", default=(160.0, 80.0), type=click.FLOAT,
    multiple=True, help="Vacuum width in angstroms",
)
@click.option(
    "--defect", "defect_list", default=(0,), type=click.INT,
    multiple=True, help="Defect count in a single cluster",
)
@click.option(
    "--seed", "random_seed", type=click.INT,
    help="Random seed",
)
@click.option(
    "--run", "run_steps", default=1000000, type=click.INT,
    help="Number of steps to run",
)
@click.option(
    "--submit", "submit_every", default=10000, type=click.INT,
    help="Number of submitting interval in steps",
)
@click.option(
    "--parallel", "parallel_workers", default=4, type=click.INT,
    help="Number of parallel workers",
)
@click.option(
    "--gpu", "gpus", type=click.INT, multiple=True,
    help="Available GPU indices",
)
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
@click.option(
    "--layers", "n_layers", default=12, type=click.INT,
    help="Number of atom layers",
)
@click.option(
    "--repeat", "repeat", default=7, type=click.INT,
    help="Number of horizontal repliation",
)
@click.option(
    "--conf", "conf", type=click.INT,
    help="Number of configurations per ndef",
)
def run_random(
    temperature: float,
    platform: Platform,
    vacuum_list: tp.Sequence[float],
    defect_list: tp.Sequence[int],
    random_seed: int,
    run_steps: int,
    submit_every: int,
    conf: int,
    parallel_workers: int,
    gpus: tp.Sequence[int],
    working_directory: Path,
    n_layers: int,
    repeat: int,
) -> None:
    def params_generator() -> tp.Iterable[Params]:
        random.seed(random_seed)
        for i, (v, d) in enumerate(product(vacuum_list, defect_list)):
            for c in range(conf):
                seed = random.getrandbits(16)
                yield Params(
                    gpu_idx=gpus[(i * conf + c) % len(gpus)],
                    vacuum=v,
                    temperature=temperature,
                    repeatition=(repeat,repeat,1),
                    n_layers=n_layers,
                    cluster=RandomCluster(d, seed, defect_type=DefectType.VACCANCY),
                    subdir=f"{v}-{d}-{seed:04x}",
                )

    with Parallel(parallel=parallel_workers, delay=1.0) as parallel:
        parallel.map(
            RunWorker(
                platform=platform,
                run_steps=run_steps,
                mean_every=None,
                submit_every=submit_every,
                working_path=working_directory,
                guard=parallel.guard,
            ),
            params_generator()
        )


if __name__ == "__main__":
    cli()
