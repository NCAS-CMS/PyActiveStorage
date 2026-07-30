from .active import Active

from importlib.metadata import version
from importlib.metadata import PackageNotFoundError


try:
    __version__ = version("PyActiveStorage")
except PackageNotFoundError as exc:
    msg = (
        "PyActiveStorage package not found, please run `pip install -e .` before "
        "importing the package."
    )
    raise PackageNotFoundError(
        msg,
    ) from exc
