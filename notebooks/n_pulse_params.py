from qml_essentials.model import Model
from rich.progress import track
from plotly import graph_objects as go

n_qubits = 10
n_layers = 10
circuit = "Circuit_18"

fig = go.Figure()

n_params = []
for q in track(range(1, n_qubits)):
    model = Model(n_qubits=q, n_layers=1, circuit_type=circuit)
    n_params.append(model.pulse_params.size)

fig.add_trace(go.Scatter(x=list(range(1, n_qubits)), y=n_params, name="Qubits"))

n_params = []
for l in track(range(1, n_layers)):
    model = Model(n_qubits=2, n_layers=l, circuit_type=circuit)
    n_params.append(model.pulse_params.size)

fig.add_trace(go.Scatter(x=list(range(1, n_qubits)), y=n_params, name="Layers"))

fig.update_layout(
    title="Number of pulse parameters",
    xaxis_title="Number of qubits",
    yaxis_title="Number of pulse parameters",
    template="plotly_white",
)
fig.show()
