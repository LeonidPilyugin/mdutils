from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import pandas as pd
import sklearn as sk
from matplotlib import pyplot as plt

from .plot import find_min_curve
from .system import DefectType
from ...plot.styles import style_context
from ...utils.timer import Timer


@click.group()
def cli() -> None:
    pass


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
    "--cogesion-energy", "cogesion_energy", type=click.FLOAT, default=98.989,
    help="Plot file name"
)
@click.option(
    "--defect-type", "defect_type",
    default=DefectType.VACCANCY, type=click.Choice(DefectType),
    help="Defect type",
)
def fit_linear_regression(
    working_directory: Path,
    data_filename: str,
    cogesion_energy: float,
    defect_type: DefectType,
):

    if defect_type is DefectType.ADATOM:
        cogesion_energy *= -1.0
    df = find_min_curve(
        pd.read_csv(working_directory / data_filename),
        cogesion_energy,
    )

    X = df["b", "n"]
    y = df["e"]

    reg = sk.linear_mode.LinearRegression(fit_intercept=False).fit(X, y)

    rmse_train = np.sqrt(sk.metrics.mean_squared_error(
        y, reg.predict(X)
    ))

    print(f"RMSE is {rmse_train:.3f} eV")

if __name__ == "__main__":
    with Timer(), style_context():
        cli()
