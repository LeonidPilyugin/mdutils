import click
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from pathlib import Path
import typing as tp
from ...plot.styles import style_context
from .system import SpiralCluster, Uo2SlabSystem, DefectType

def find_min_curve(
    df: pd.DataFrame,
    cogesion_energy: float,
    reference_energy: float | None = None,
) -> pd.DataFrame:
    ddf = df.groupby(["n", "b"], as_index=False)["e"].min()
    assert isinstance(ddf, pd.DataFrame)

    if reference_energy is None:
        reference_energy = ddf.loc[ddf["n"] == 0, "e"].iloc[0]

    ddf["e"] -= ddf["n"] * cogesion_energy + reference_energy
    return ddf


@click.group()
def cli() -> None:
    pass


@cli.command("e2n")
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
@click.option(
    "--data", "data_filename", type=click.STRING, default="minimize.csv",
    help="Data file name"
)
@click.option(
    "--plot", "plot_filename", type=click.STRING, default="minimize.png",
    help="Plot file name"
)
def plot_energy_vs_n(
    working_directory: Path,
    data_filename: str,
    plot_filename: str,
) -> None:
    """Plot energy vs number of defects in single cluster"""
    plt.cla()
    df = pd.read_csv(working_directory / data_filename)
    a, b = np.polyfit(df["n"], df["e"], 1)

    plt.plot(df["n"], df["e"], ".")

    plt.axline((0, b), slope=a, color="red", linestyle="--",
               label=f"LSQ: y={a:.2f}x+{b:.2f}")
    plt.legend()

    plt.xlabel("N")
    plt.ylabel("E, eV")

    plt.savefig(working_directory / plot_filename)


@cli.command("de2n")
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
@click.option(
    "--data", "data_filename", type=click.STRING, default="minimize.csv",
    help="Data file name"
)
@click.option(
    "--plot", "plot_filename", type=click.STRING, default="minimize.png",
    help="Plot file name"
)
@click.option(
    "--cogesion-energy", "cogesion_energy", type=click.FLOAT, default=98.989,
    help="Plot file name"
)
@click.option(
    "--defect-type", "defect_type",
    default=DefectType.VACCANCY, type=click.Choice(DefectType),
    help="Defect type",
)
def plot_de_vs_n(
    working_directory: Path,
    data_filename: str,
    plot_filename: str,
    cogesion_energy: float,
    defect_type: DefectType,
) -> None:
    """Plot energy difference vs number of defects in single cluster"""

    if defect_type is DefectType.VACCANCY:
        cogesion_energy *= -1.0

    plt.cla()
    df = find_min_curve(
        pd.read_csv(working_directory / data_filename),
        cogesion_energy,
    )
    a, b = np.polyfit(df["n"], df["e"], 1)

    plt.plot(df["n"], df["e"], ".")

    plt.axline((0, b), slope=a, color="red", linestyle="--",
               label=f"LSQ: y={a:.2f}x+{b:.2f}")
    plt.legend()

    plt.xlabel("N")
    plt.ylabel("E, eV")

    plt.savefig(working_directory / plot_filename)


@cli.command("de2b")
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
@click.option(
    "--data", "data_filenames", type=click.STRING, default="minimize.csv",
    multiple=True, help="Data file name"
)
@click.option(
    "--label", "labels", type=click.STRING, default="",
    multiple=True, help="Data file name"
)
@click.option(
    "--plot", "plot_filename", type=click.STRING, default="minimize.png",
    help="Plot file name"
)
@click.option(
    "--cogesion-energy", "cogesion_energy", type=click.FLOAT, default=98.989,
    help="Plot file name"
)
@click.option(
    "--defect-type", "defect_type", default=DefectType.VACCANCY,
    type=click.Choice(DefectType),
    help="Defect type",
)
def plot_de_vs_b(
    working_directory: Path,
    data_filenames: tp.Tuple[str],
    plot_filename: str,
    cogesion_energy: float,
    defect_type: DefectType,
    labels: tp.Tuple[str],
) -> None:
    """Plot energy difference vs number of cut horizontal bonds"""

    if defect_type is DefectType.ADATOM:
        cogesion_energy *= -1.0

    assert len(labels) == len(data_filenames)

    plt.cla()

    data_filename = data_filenames[0]

    df = find_min_curve(
        pd.read_csv(working_directory / data_filename),
        cogesion_energy,
    )
    a, b = np.polyfit(df["b"], df["e"], 1)

    plt.axline((0, b), slope=a, color="red", linestyle="--",
               label=f"LSQ: y={a:.2f}x+{b:.2f}")

    for label, data_filename in zip(labels, data_filenames):
        df = find_min_curve(
            pd.read_csv(working_directory / data_filename),
            cogesion_energy,
        )
        plt.plot(df["b"], df["e"], ".", label=label)

    plt.legend()

    plt.xlabel("N_{bonds}")
    plt.ylabel("E, eV")

    plt.savefig(working_directory / plot_filename)

@cli.command("n2b")
@click.option(
    "--wdir", "working_directory",
    type=click.Path(path_type=Path, exists=False, file_okay=False,
                    writable=True),
    help="Working directory"
)
@click.option(
    "--data", "data_filename", type=click.STRING, default="minimize.csv",
    help="Data file name"
)
@click.option(
    "--plot", "plot_filename", type=click.STRING, default="minimize.png",
    help="Plot file name"
)
def plot_vac_vs_bonds(
    working_directory: Path,
    data_filename: str,
    plot_filename: str,
) -> None:
    """Plot total number of cut bonds vs total number of defects"""
    plt.cla()

    df = pd.read_csv(working_directory / data_filename)
    plt.scatter(df["n"] * 2, df["b"])

    plt.xlabel("N")
    plt.ylabel("N_{bonds}")

    plt.savefig(working_directory / plot_filename)


if __name__ == "__main__":
    with style_context():
        cli()
