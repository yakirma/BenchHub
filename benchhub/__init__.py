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
    get_type,
)
from benchhub.client import (
    BenchHubAPIError,
    BHDatasetCreator,
    Client,
    FlaskTestClientTransport,
    SubmissionBuilder,
)

__all__ = [
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
    "get_type",
    "Client",
    "SubmissionBuilder",
    "BHDatasetCreator",
    "FlaskTestClientTransport",
    "BenchHubAPIError",
]
