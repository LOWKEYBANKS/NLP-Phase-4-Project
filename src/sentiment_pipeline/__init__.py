from importlib import import_module

# Re-export public API from .pipeline
_pipeline = import_module("sentiment_pipeline.pipeline")

get_processed_dataframe = _pipeline.get_processed_dataframe  # noqa: F401
prepare_dataframe = _pipeline.prepare_dataframe  # noqa: F401
load_raw_data = _pipeline.load_raw_data  # noqa: F401

__all__ = [
    "get_processed_dataframe",
    "prepare_dataframe",
    "load_raw_data",
]