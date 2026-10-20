from contextlib import contextmanager
from importlib.resources import as_file, files

import matplotlib.pyplot as plt


def use_style(name="regular"):
    """Apply a stylesheet."""
    resource = files("mdutils").joinpath(
        "plot", "styles", f"{name}.mplstyle"
    )

    with as_file(resource) as path:
        plt.style.use(str(path))


@contextmanager
def style_context(name="regular"):
    """Temporarily apply stylesheet."""
    resource = files("mdutils").joinpath(
        "plot", "styles", f"{name}.mplstyle"
    )

    with as_file(resource) as path:
        with plt.style.context(str(path)):
            yield

