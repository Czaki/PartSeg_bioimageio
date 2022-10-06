try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from ._settings import get_settings

try:
    # noinspection PyUnresolvedReferences,PyUnboundLocalVariable
    reloading
except NameError:
    reloading = False
else:
    reloading = True

__all__ = ("get_settings", "register")


def register():
    from PartSegCore.register import RegisterEnum
    from PartSegCore.register import register as register_fun

    from . import segmentation_algorithm

    if reloading:
        import importlib

        importlib.reload(segmentation_algorithm)

    register_fun(
        segmentation_algorithm.BioimageioROIExtraction,
        RegisterEnum.analysis_algorithm,
    )
