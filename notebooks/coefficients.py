import marimo

__generated_with = "0.19.7"
app = marimo.App()


@app.cell
def _():
    from qml_essentials.model import Model
    from qml_essentials.coefficients import Coefficients

    import pennylane.numpy as np

    import plotly.graph_objects as go

    model = Model(
        n_qubits=4,
        n_layers=2,
        circuit_type="Circuit_15",
        encoding="RY"
    )

    model.initialize_params(rng=np.random.default_rng(1000), repeat=1000)

    coeffs, freqs = Coefficients.get_spectrum(model, shift=True, trim=True)


    return coeffs, freqs, go, np


@app.cell
def _(coeffs, freqs, go, np):
    import marimo as mo

    fig = go.Figure()
    print(coeffs.shape)
    variance = np.abs(coeffs).mean(axis=1)
    fig.add_scatter(
        x = freqs,
        y = variance
    )
    fig.update_layout(template="plotly_white")

    plot = mo.ui.plotly(fig)
    mo.hstack([plot])
    return


if __name__ == "__main__":
    app.run()
