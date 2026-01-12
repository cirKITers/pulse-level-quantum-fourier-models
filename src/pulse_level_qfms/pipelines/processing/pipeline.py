from kedro.pipeline import Node, Pipeline

from pulse_level_qfms.pipelines.processing.nodes import calculate_fcc, train_model


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                train_model,
                name="train_model",
                tags=["processing"],
                inputs=[
                    "model",
                    "train_loader",
                    "valid_loader",
                    "params:model.noise_params",
                    "params:train.loss_functions",
                    "params:train.loss_scalers",
                    "params:train.steps",
                    "params:train.learning_rate",
                ],
                outputs={
                    "model": "trained_model",
                },
            ),
            Node(
                calculate_fcc,
                name="calculate_fcc",
                tags=["processing"],
                inputs=[
                    "trained_model",
                    "params:fcc.seed",
                    "params:fcc.n_samples",
                    "params:model.noise_params",
                ],
                outputs={
                    "fcc": "fcc",
                },
            ),
        ]
    )
