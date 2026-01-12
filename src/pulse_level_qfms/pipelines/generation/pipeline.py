from kedro.pipeline import Node, Pipeline

from pulse_level_qfms.pipelines.generation.nodes import (
    generate_model,
    generate_fourier_series,
    build_fourier_series_dataloader,
)


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline(
        [
            Node(
                generate_model,
                name="generate_model",
                tags=["generation"],
                inputs=[
                    "params:model.n_qubits",
                    "params:model.n_layers",
                    "params:model.circuit_type",
                    "params:model.data_reupload",
                    "params:model.encoding_gates",
                    "params:model.encoding_strategy",
                    "params:model.initialization",
                    "params:model.initialization_domain",
                    "params:model.output_qubit",
                    "params:model.mp_threshold",
                    "params:model.seed",
                ],
                outputs={
                    "model": "model",
                },
            ),
            Node(
                generate_fourier_series,
                name="generate_fourier_series",
                tags=["generation"],
                inputs=[
                    "model",
                    "params:data.domain",
                    "params:data.omegas",
                    "params:data.coefficients_min",
                    "params:data.coefficients_max",
                    "params:data.zero_centered",
                    "params:data.seed",
                ],
                outputs={
                    "domain_samples": "domain_samples",
                    "fourier_samples": "fourier_samples",
                    "coefficients": "coefficients",
                },
            ),
            Node(
                build_fourier_series_dataloader,
                name="build_fourier_series_dataloader",
                tags=["generation"],
                inputs=[
                    "params:data.batch_size",
                    "domain_samples",
                    "fourier_samples",
                    "coefficients",
                ],
                outputs={
                    "train_loader": "train_loader",
                    "valid_loader": "valid_loader",
                },
            ),
        ]
    )
