try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"


from ._settings import get_settings

__all__ = ("get_settings",)
