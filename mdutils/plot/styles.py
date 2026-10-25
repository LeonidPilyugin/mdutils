import typing as tp
from contextlib import contextmanager
from importlib.resources import as_file, files

import matplotlib.pyplot as plt


def use_style(name: str = "regular") -> None:
    """Apply a stylesheet."""
    resource = files("mdutils").joinpath(
        "plot", "styles", f"{name}.mplstyle"
    )

    with as_file(resource) as path:
        plt.style.use(str(path))


@contextmanager
def style_context(name: str = "regular") -> tp.Generator[None]:
    """Temporarily apply stylesheet."""
    resource = files("mdutils").joinpath(
        "plot", "styles", f"{name}.mplstyle"
    )

    with as_file(resource) as path, plt.style.context(str(path)):
        yield

