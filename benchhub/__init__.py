"""BenchHub — shared types + client API.

This package is imported by the server (`app.py`, `metric_engine.py`) AND
shipped to submitters as `benchhub-client`. Single source of truth for
the strict-typed contract between predictions, GT, and metrics.
"""

from benchhub.types import (
    DataType,
    DTYPES,
    Image,
    Mask,
    Depth,
    Audio,
    Text,
    BBoxes,
    Label,
    LabelList,
    Scalar,
    Json,
    Sequence,
    CocoDetections,
    get_type,
)
from benchhub.client import (
    BenchHubAPIError,
    BHDatasetCreator,
    Client,
    FlaskTestClientTransport,
    RawPrediction,
    SubmissionBuilder,
)
from benchhub import author

__all__ = [
    "author",
    "DataType",
    "DTYPES",
    "Image",
    "Mask",
    "Depth",
    "Audio",
    "Text",
    "BBoxes",
    "Label",
    "LabelList",
    "Scalar",
    "Json",
    "Sequence",
    "CocoDetections",
    "get_type",
    "Client",
    "SubmissionBuilder",
    "RawPrediction",
    "BHDatasetCreator",
    "FlaskTestClientTransport",
    "BenchHubAPIError",
]
