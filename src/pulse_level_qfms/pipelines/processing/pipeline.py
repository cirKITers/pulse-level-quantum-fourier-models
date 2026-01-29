from kedro.pipeline import Node, Pipeline

from pulse_level_qfms.pipelines.processing.nodes import (
    calculate_fcc,
    train_model,
    evaluate_fidelity,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            # Node(
            #     train_model,
            #     name="train_model",
            #     tags=["processing"],
            #     inputs=[
            #         "model",
            #         "train_loader",
            #         "valid_loader",
            #         "params:model.noise_params",
            #         "params:train.loss_functions",
            #         "params:train.loss_scalers",
            #         "params:train.steps",
            #         "params:train.learning_rate",
            #     ],
            #     outputs={
            #         "model": "trained_model",
            #     },
            # ),
            Node(
                calculate_fcc,
                name="calculate_fcc",
                tags=["processing"],
                inputs=[
                    "model",
                    "params:fcc.seed",
                    "params:fcc.n_samples",
                    "params:fcc.sample_axis",
                    "params:fcc.pulse_params_variance",
                ],
                outputs={
                    "fcc": "fcc",
                },
            ),
            # Node(
            #     evaluate_fidelity,
            #     name="evaluate_fidelity",
            #     tags=["processing"],
            #     inputs=[
            #         "model",
            #         "params:fcc.seed",
            #         "params:fcc.n_samples",
            #         "params:fcc.pulse_params_variance",
            #     ],
            #     outputs={
            #         "fidelity": "fidelity",
            #     },
            # ),
        ]
    )
