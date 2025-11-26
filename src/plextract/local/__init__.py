"""
Local pipeline for chart data extraction.

Requires the [local] extra to be installed:
    pip install plextract[local]
"""


def __getattr__(name: str):
    """Lazy import to avoid loading heavy ML dependencies at module import time."""
    if name == "run_pipeline":
        from .app import run_pipeline
        return run_pipeline
    elif name == "LineFormer":
        from .lineformer import LineFormer
        return LineFormer
    elif name == "ChartDete":
        from .chartdete import ChartDete
        return ChartDete
    elif name == "OCRModel":
        from .trocr import OCRModel
        return OCRModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["run_pipeline", "LineFormer", "ChartDete", "OCRModel"]

