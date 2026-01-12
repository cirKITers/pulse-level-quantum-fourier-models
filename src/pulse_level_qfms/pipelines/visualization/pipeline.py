from kedro.pipeline import Node, Pipeline

from pulse_level_qfms.pipelines.visualization.nodes import (
    visualize_time_domain,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                visualize_time_domain,
                name="visualize_time_domain",
                tags=["visualization"],
                inputs=[
                    "trained_model",
                    "train_loader",
                    "params:model.noise_params",
                ],
                outputs={
                    "figure": "fig_time_domain",
                },
            )
        ]
    )
