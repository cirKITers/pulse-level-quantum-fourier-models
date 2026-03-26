"""Project pipelines."""

from __future__ import annotations

from kedro.framework.project import find_pipelines
from kedro.pipeline import Pipeline

from .pipelines.generation.pipeline import (
    create_data_pipeline as generation_data_pipeline,
)
from .pipelines.generation.pipeline import (
    create_model_pipeline as generation_model_pipeline,
)
from .pipelines.processing.pipeline import (
    create_fcc_pipeline as processing_fcc_pipeline,
)
from .pipelines.processing.pipeline import (
    create_fidelity_pipeline as processing_fidelity_pipeline,
)
from .pipelines.processing.pipeline import (
    create_expressibility_pipeline as processing_expressibility_pipeline,
)
from .pipelines.processing.pipeline import (
    create_training_pipeline as processing_training_pipeline,
)
from .pipelines.visualization.pipeline import (
    create_training_pipeline as visualization_training_pipeline,
)


def register_pipelines() -> dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    pipelines = find_pipelines()
    pipelines["__default__"] = sum(pipelines.values())

    pipelines["study-1"] = generation_model_pipeline() + processing_fcc_pipeline()
    pipelines["study-2"] = generation_model_pipeline() + processing_fidelity_pipeline()
    pipelines["study-3"] = (
        generation_model_pipeline() + processing_expressibility_pipeline()
    )
    pipelines["study-4"] = (
        generation_data_pipeline()
        + generation_model_pipeline()
        + processing_training_pipeline()
        + visualization_training_pipeline()
    )

    return pipelines
