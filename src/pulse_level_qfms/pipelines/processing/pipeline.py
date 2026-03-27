from kedro.pipeline import Node, Pipeline

from pulse_level_qfms.pipelines.processing.nodes import (
    calculate_fcc,
    train_model,
    evaluate_fidelity,
    evaluate_expressibility,
)


def create_fcc_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                calculate_fcc,
                name="calculate_fcc",
                tags=["processing"],
                inputs=[
                    "model",
                    "params:fcc.seed",
                    "params:fcc.n_samples",
                    "params:fcc.scale",
                    "params:fcc.method",
                    "params:fcc.weighting",
                    "params:fcc.sample_axis",
                    "params:fcc.pulse_params_variance",
                    "params:fcc.numerical_cap",
                ],
                outputs={
                    "fcc": "fcc",
                },
            ),
        ]
    )


def create_fidelity_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                evaluate_fidelity,
                name="evaluate_fidelity",
                tags=["processing"],
                inputs=[
                    "model",
                    "params:fidelity.seed",
                    "params:fidelity.n_samples",
                    "params:fidelity.scale",
                    "params:fidelity.pulse_params_variance",
                ],
                outputs={
                    "fidelity": "fidelity",
                },
            ),
        ]
    )


def create_expressibility_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                evaluate_expressibility,
                name="evaluate_expressibility",
                tags=["processing"],
                inputs=[
                    "model",
                    "params:expressibility.seed",
                    "params:expressibility.n_samples",
                    "params:expressibility.n_bins",
                    "params:expressibility.scale",
                    "params:expressibility.pulse_params_variance",
                ],
                outputs={
                    "fidelity": "fidelity",
                },
            ),
        ]
    )


def create_training_pipeline(**kwargs) -> Pipeline:
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
                    "params:train.train_axis",
                    "params:train.pulse_learning_rate",
                ],
                outputs={
                    "model": "trained_model",
                },
            ),
        ]
    )
