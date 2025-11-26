"""
Modal-based pipeline for chart data extraction (runs on Modal cloud).

Requires the [modal] extra to be installed:
    pip install plextract[modal]
"""


def __getattr__(name: str):
    """Lazy import to avoid loading modal at module import time."""
    if name == "run_pipeline":
        from .app import run_pipeline
        return run_pipeline
    elif name == "modal_app":
        from .modal import modal_app
        return modal_app
    elif name == "vol":
        from .modal import vol
        return vol
    elif name == "download_volume_dir":
        from .modal import download_volume_dir
        return download_volume_dir
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ['modal_app', 'vol', 'download_volume_dir', 'run_pipeline']