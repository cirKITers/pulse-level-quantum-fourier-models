import os
import re
import plotly
import plotly.graph_objects as go
from typing import List
import numpy as np
import pandas as pd
import string
from plotly.subplots import make_subplots
from scipy.signal import argrelmin, argrelmax


def _natural_sort_key(s: str):
    """Sort key that handles embedded numbers naturally.
    E.g. Circuit_3 < Circuit_8 < Circuit_13 instead of lexicographic order."""
    return [
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", s)
    ]


def sort_ansatzes(ansatzes):
    """Sort ansatz names using natural ordering."""
    return sorted(ansatzes, key=_natural_sort_key)


def _collect_metric_history(df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Build a wide DataFrame of a per-step metric history.

    Reads the list-valued columns ``<metric_name>.steps`` and
    ``<metric_name>.values`` from the rows of ``df`` (one row per run) and
    returns a DataFrame indexed by step with one column per run_id.  Rows
    that do not carry the metric are skipped.  If no row carries it, an
    empty DataFrame is returned.
    """
    steps_col = f"{metric_name}.steps"
    values_col = f"{metric_name}.values"

    if steps_col not in df.columns or values_col not in df.columns:
        return pd.DataFrame()

    series_by_run = {}
    for _, row in df.iterrows():
        steps = row.get(steps_col)
        values = row.get(values_col)
        # List-valued columns may be NaN for runs that do not log this metric
        if not isinstance(steps, (list, tuple, np.ndarray)):
            continue
        if not isinstance(values, (list, tuple, np.ndarray)):
            continue
        if len(steps) == 0:
            continue
        series_by_run[row["run_id"]] = pd.Series(
            data=list(values), index=list(steps)
        )

    if not series_by_run:
        return pd.DataFrame()

    return pd.DataFrame(series_by_run).sort_index()


class design:
    template = "plotly_white"
    font_size = 22
    marker_size = 14
    marker_line_width = 2
    marker_a_opacity = 1.0
    marker_b_opacity = 1.0
    marker_a_style = "x"
    marker_a_color = "#009682"
    marker_b_style = "x"
    marker_b_color = "#DF9B1B"
    legend_color = "#002D4C"
    colorscale = "Sunset"
    symbols_lst = [
        "circle",
        "square",
        "diamond",
        "cross",
        "x",
        "triangle-up",
        "hexagon",
        "star",
        "star-square",
        "y-up",
        "bowtie",
        "hourglass",
        "cross-thin",
    ]
    prim_colors_lst = plotly.colors.qualitative.Vivid
    sec_colors_lst = plotly.colors.qualitative.Safe
    seq_colors = plotly.colors.sequential.dense_r

    @staticmethod
    def horizontal_legend():
        """Returns legend configuration for horizontal layout below figure."""
        return dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        )


def circuit_name_to_str(circuit_name: str):
    if "Circuit" in circuit_name:
        circuit_name = circuit_name.replace("Circuit", "C")
    elif "Hardware_Efficient" in circuit_name:
        circuit_name = "HEA"
    elif "Strongly_Entangling" in circuit_name:
        circuit_name = "SEA"

    circuit_name = circuit_name.replace("_", "")

    return circuit_name


def save_figures(
    figures: List[go.Figure],
    name: str,
    experiment_id: str,
    hash: str,
    scale: float = 1,
):
    # use the same hashing strategy as in data_helper
    path = f"results/{experiment_id}/{hash}/"
    os.makedirs(path, exist_ok=True)
    abc = list(string.ascii_lowercase)
    for it, fig in enumerate(figures):
        # applying last changes
        fig.update_layout()

        filename = f"{path}{name}-{abc[it]}.pdf"
        print(f"Saving figure to {filename}")
        fig.write_image(filename, scale=scale)


def viz_study_1(df, max_distortion, threshold, show_error):
    figures = []

    figures.append(fcc_over_distortion(df, max_distortion, show_error))
    # figures.append(coeff_mean_over_distortion(df, max_distortion, show_error))
    # figures.append(coeff_var_over_distortion(df, max_distortion, show_error))
    figures.append(coeff_var_delta_over_distortion(df, max_distortion, show_error))
    figures.append(
        frequency_histogram_by_distortion(df, max_distortion, threshold, show_error)
    )

    return figures


def viz_study_2(df, max_distortion, show_error):
    figures = []

    figures.append(fidelity_over_distortion(df, max_distortion, show_error))
    figures.append(trace_distance_over_distortion(df, max_distortion, show_error))

    return figures


def viz_study_3(df, max_distortion, show_error):
    figures = []

    figures.append(expressibility_over_distortion(df, max_distortion, show_error))

    return figures


def viz_study_4(df, show_error):
    figures = []

    figures.append(
        pulse_param_mse_comparison(df, show_error)
    )
    figures.extend(pulse_mean_and_variance_over_step(df, show_error))
    figures.append(loss_over_step(df, show_error))

    return figures


def viz_study_5(df, max_distortion, show_error):
    figures = []

    figures.extend(spectrum_over_distortion(df, max_distortion, show_error))
    figures.append(offgrid_mass_over_distortion(df, max_distortion, show_error))

    return figures


def viz_study_6(df):
    figures = []

    figures.append(landscape_over_eta(df, "profile"))
    figures.append(landscape_over_eta(df, "fixed"))
    figures.append(landscape_scaling(df))
    figures.append(landscape_scaling_frequencies(df))
    figures.append(landscape_over_circuits(df))

    return figures


def _landscape_row(df: pd.DataFrame) -> pd.Series:
    """Return the first row of `df` that carries a landscape sweep.

    Landscapes are not averaged over runs: every run draws its own target
    scalers, so the curves of two runs are minima at different places.
    """
    for _, row in df.iterrows():
        if any(
            c.startswith("landscape.eta.")
            and c.endswith(".values")
            and isinstance(row.get(c), (list, tuple, np.ndarray))
            for c in row.index
        ):
            return row

    raise ValueError("No landscape sweep in this DataFrame, run study-6 first.")


def _landscape_gates(row: pd.Series) -> List[str]:
    """The encoding gates swept in `row`, as their ``l<layer>.q<qubit>`` keys.

    Ordered by the generator the gate drives, so the panels of a figure run
    from the lowest spectral component to the highest.
    """
    # once runs of different sizes share a DataFrame, every row carries the
    # columns of the widest one, so the gates of this run are the ones whose
    # sweep actually holds values
    keys = [
        c[len("landscape.eta.") : -len(".values")]
        for c in row.index
        if c.startswith("landscape.eta.")
        and c.endswith(".values")
        and isinstance(row[c], (list, tuple, np.ndarray))
    ]

    return sorted(keys, key=lambda k: (row[f"landscape.generator.{k}"], k))


def _gate_label(row: pd.Series, key: str, unique: bool) -> str:
    """Trace name of an encoding gate, by the generator it drives.

    Two gates can share a generator, across layers or, under hamming, across
    qubits as well. Their position is only spelled out when it is needed to
    tell them apart.
    """
    generator = row[f"landscape.generator.{key}"]
    if unique:
        return f"$\\gamma = {generator:g}$"

    layer, qubit = key.split(".")
    return f"$\\gamma = {generator:g}$ ({layer}{qubit})"


def landscape_over_eta(df: pd.DataFrame, curve: str):
    """
    Plot the loss over the encoding scaler, one panel per encoding gate.

    Each panel is a slice through the loss in which a single encoding scaler
    is swept while the others sit at their target values, so the slice runs
    from the initial scaler $\\eta = 1$ through the aligned one. The
    oscillation along $\\eta$ has period $1/(mts \\cdot \\gamma)$ for a gate
    driving the generator $\\gamma$, so the panels of the higher generators
    pack proportionally more local minima between the two. The panels share the
    scaler axis but not the loss axis, which differs by orders of magnitude
    between them.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``generate_df``.
        curve (str): Which loss to plot, "profile" for the loss with the
            coefficients concentrated out or "fixed" for the loss at the
            current variational parameters. The profile panels overlay the
            closed-form Dirichlet approximation when the run logged it.

    Returns:
        go.Figure: A figure showing the loss over the encoding scaler.
    """
    # the concentrated loss reaches machine zero where the comb covers the
    # target, which a log axis cannot show. Values below this floor are drawn
    # at the floor.
    floor = 1e-9

    row = _landscape_row(df)
    gates = _landscape_gates(row)
    generators = [row[f"landscape.generator.{key}"] for key in gates]
    unique = len(set(generators)) == len(generators)

    fig = make_subplots(
        rows=len(gates),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.15 / len(gates),
    )
    color_it = iter(design.prim_colors_lst)

    for it, key in enumerate(gates):
        grid = np.array(row[f"landscape.eta.{key}.values"])
        values = np.array(row[f"landscape.{curve}.{key}.values"])
        target = row[f"landscape.target_eta.{key}"]
        color = next(color_it)

        fig.add_scatter(
            x=grid,
            y=np.clip(values, floor, None),
            mode="lines",
            name=_gate_label(row, key, unique),
            line=dict(color=color, width=1.5),
            row=it + 1,
            col=1,
        )

        analytic = row.get(f"landscape.analytic.{key}.values")
        if curve == "profile" and isinstance(analytic, (list, tuple, np.ndarray)):
            fig.add_scatter(
                x=grid,
                y=np.clip(np.array(analytic), floor, None),
                mode="lines",
                name="analytic",
                line=dict(color="gray", width=1, dash="dash"),
                showlegend=it == 0,
                row=it + 1,
                col=1,
            )

        fig.add_vline(
            x=target,
            line=dict(color=color, width=1.5, dash="dot"),
            row=it + 1,
            col=1,
        )
        fig.add_vline(
            x=1.0,
            line=dict(color=design.legend_color, width=1.5, dash="dash"),
            row=it + 1,
            col=1,
        )
        fig.update_yaxes(title_text="MSE", type="log", row=it + 1, col=1)

    fig.update_xaxes(title_text="$\\eta$", row=len(gates), col=1)
    fig.update_layout(
        title=("Concentrated" if curve == "profile" else "Fixed-parameter")
        + " Loss over Encoding Scaler",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
        margin=dict(b=120),
        height=300 * len(gates),
    )

    return fig


def _basin_and_minima(grid: np.ndarray, values: np.ndarray, target: float):
    """Basin width around the aligned scaler and density of local minima on the
    way to it.

    The basin is the span between the two local maxima flanking the minimum at
    `target`, i.e. the main lobe of the loss. The density is the number of
    local minima strictly between the initial scaler $\\eta = 1$ and `target`
    per unit of scaler travelled, which is how often a descent from the initial
    scaler can stall. Dividing by the path length removes the random draw of
    the target scaler, leaving a quantity that depends on the generator alone.

    Args:
        grid (np.ndarray): The scaler grid.
        values (np.ndarray): Loss over the grid.
        target (float): The aligned scaler.

    Returns:
        tuple[float, float]: Basin width and local minima per unit scaler.
    """
    center = int(np.argmin(np.abs(grid - target)))
    maxima = argrelmax(values)[0]

    left = maxima[maxima < center]
    right = maxima[maxima > center]
    width = grid[right[0]] - grid[left[-1]] if len(left) and len(right) else np.nan

    inside = (grid >= min(1.0, target)) & (grid <= max(1.0, target))
    distance = abs(target - 1.0)
    density = len(argrelmin(values[inside])[0]) / distance if distance else np.nan

    return float(width), float(density)


def landscape_scaling(df: pd.DataFrame):
    """
    Plot the basin width and the density of local minima against the generator
    the swept gate drives, one marker per gate and run.

    Both follow from the Dirichlet kernel of the sample window: the loss
    oscillates with period $1/(mts \\cdot \\gamma)$ along the scaler of a gate
    driving the generator $\\gamma$, so the main lobe spans
    $2/(mts \\cdot \\gamma)$ and the path from the initial to the aligned
    scaler crosses $mts \\cdot \\gamma$ oscillations per unit of scaler
    travelled. The measured values are read off the concentrated loss, which
    depends on the encoding and the target only, so the spread across markers
    at one generator reflects the target draw rather than the ansatz.

    The basin reference is the sharper of the two: counting strict local minima
    over a path only a few oscillations long is a coarse statistic, and the
    superposition over target components merges part of the lobes.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``generate_df``.

    Returns:
        go.Figure: A figure showing both quantities over the generator.
    """
    mts = None
    generators, widths, densities = [], [], []
    for row in _landscape_rows(df):
        mts = row["landscape.mts"]
        for key in _landscape_gates(row):
            width, density = _basin_and_minima(
                np.array(row[f"landscape.eta.{key}.values"]),
                np.array(row[f"landscape.profile.{key}.values"]),
                row[f"landscape.target_eta.{key}"],
            )
            generators.append(row[f"landscape.generator.{key}"])
            widths.append(width)
            densities.append(density)

    generators = np.array(generators)
    reference = np.unique(generators)

    # the densities reach zero where the aligned scaler still sits inside the
    # initial basin, so they get a linear axis of their own while the basin
    # widths keep the log axis their power law needs
    fig = go.Figure()
    fig.add_scatter(
        x=generators,
        y=densities,
        mode="markers",
        name="local minima",
        marker=dict(color=design.prim_colors_lst[0], size=design.marker_size),
    )
    fig.add_scatter(
        x=reference,
        y=mts * reference,
        mode="lines",
        name="$mts \\cdot \\gamma$",
        line=dict(color=design.prim_colors_lst[0], width=1.5, dash="dash"),
    )
    fig.add_scatter(
        x=generators,
        y=widths,
        mode="markers",
        name="basin width",
        yaxis="y2",
        marker=dict(color=design.prim_colors_lst[1], size=design.marker_size),
    )
    fig.add_scatter(
        x=reference,
        y=2.0 / (mts * reference),
        mode="lines",
        name="$2 / (mts \\cdot \\gamma)$",
        yaxis="y2",
        line=dict(color=design.prim_colors_lst[1], width=1.5, dash="dash"),
    )

    fig.update_layout(
        title="Landscape Scaling over Encoding Generator",
        xaxis=dict(
            title="$\\gamma$",
            type="log",
            tickmode="array",
            tickvals=reference,
        ),
        yaxis=dict(
            title="minima per unit scaler",
            rangemode="tozero",
            color=design.prim_colors_lst[0],
        ),
        yaxis2=dict(
            title="basin width",
            type="log",
            overlaying="y",
            side="right",
            color=design.prim_colors_lst[1],
        ),
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
        margin=dict(b=160),
    )

    return fig


def _landscape_rows(df: pd.DataFrame) -> List[pd.Series]:
    """All rows of `df` that carry a landscape sweep."""
    rows = []
    for _, row in df.iterrows():
        if any(
            c.startswith("landscape.eta.")
            and c.endswith(".values")
            and isinstance(row.get(c), (list, tuple, np.ndarray))
            for c in row.index
        ):
            rows.append(row)

    if not rows:
        raise ValueError("No landscape sweep in this DataFrame, run study-6 first.")

    return rows


def _strategy_generator_max(strategy: str, n_frequencies: float) -> float:
    """Largest encoding generator of a single-layer model with the given
    spectrum size.

    Inverts the comb sizes $|\\Omega| = 3^n$, $2^{n+1} - 1$ and $2n + 1$ of
    the ternary, binary and hamming strategies to the generator of their
    highest qubit, $3^{n-1}$, $2^{n-1}$ and $1$.
    """
    if strategy == "ternary":
        return n_frequencies / 3.0
    if strategy == "binary":
        return (n_frequencies + 1) / 4.0
    return 1.0


def landscape_scaling_frequencies(df: pd.DataFrame):
    """
    Plot the hardness of the highest encoding generator against the number of
    frequencies of the model, one point per run, one trace per encoding
    strategy.

    Hardness is read off the concentrated loss of the gates driving the largest
    generator $\\gamma_{max}$, averaged over them when several do, and over the
    runs that share a spectrum size, whose spread is shown as an error bar: the
    width of
    the basin around the aligned scaler and the number of local minima per unit
    of scaler travelled towards it. The Dirichlet geometry gives both in closed
    form, $W = 2 / (mts \\cdot \\gamma_{max})$ and
    $\\nu = mts \\cdot \\gamma_{max}$, and the encoding strategy ties
    $\\gamma_{max}$ to the spectrum size $|\\Omega|$: $|\\Omega| / 3$ for
    ternary, $(|\\Omega| + 1) / 4$ for binary and $1$ for hamming. Hardness
    therefore grows linearly with the spectrum for the exponential encodings
    and stays flat for hamming, which is the control.

    The basin is the sharper of the two measurements. The minima density counts
    strict local minima over a path only a few oscillations long, so it carries
    visibly more scatter.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``generate_df``, one row per run of the
            scaling sweep.

    Returns:
        go.Figure: A figure showing both quantities over the spectrum size.
    """
    points = {}
    for row in _landscape_rows(df):
        # the generator inversion below only holds for a single layer, so
        # multi-layer runs in the same experiment are left out
        if row["landscape.n_gates"] != row["model.n_qubits"]:
            continue

        keys = _landscape_gates(row)
        generator = max(row[f"landscape.generator.{k}"] for k in keys)
        measured = [
            _basin_and_minima(
                np.array(row[f"landscape.eta.{k}.values"]),
                np.array(row[f"landscape.profile.{k}.values"]),
                row[f"landscape.target_eta.{k}"],
            )
            for k in keys
            if row[f"landscape.generator.{k}"] == generator
        ]

        points.setdefault(row["model.encoding_strategy"], []).append(
            {
                "n_frequencies": row["landscape.n_frequencies"],
                "width": np.nanmean([w for w, _ in measured]),
                "density": np.nanmean([d for _, d in measured]),
                "mts": row["landscape.mts"],
            }
        )

    fig = go.Figure()
    symbols = dict(zip(sorted(points), design.symbols_lst))

    for strategy, entries in sorted(points.items()):
        # several runs share a spectrum size once seeds are swept, so they are
        # averaged and their spread reported as an error bar
        n = np.array(sorted({e["n_frequencies"] for e in entries}))
        mts = entries[0]["mts"]
        gamma = np.array([_strategy_generator_max(strategy, v) for v in n])
        grouped = [[e for e in entries if e["n_frequencies"] == v] for v in n]
        width = np.array([np.nanmean([e["width"] for e in g]) for g in grouped])
        width_sd = np.array([np.nanstd([e["width"] for e in g]) for g in grouped])
        density = np.array([np.nanmean([e["density"] for e in g]) for g in grouped])
        density_sd = np.array([np.nanstd([e["density"] for e in g]) for g in grouped])

        fig.add_scatter(
            x=n,
            y=width,
            error_y=dict(type="data", array=width_sd, visible=True),
            mode="markers",
            name=f"basin ({strategy})",
            marker=dict(
                color=design.prim_colors_lst[1],
                size=design.marker_size,
                symbol=symbols[strategy],
            ),
        )
        fig.add_scatter(
            x=n,
            y=2.0 / (mts * gamma),
            mode="lines",
            showlegend=False,
            line=dict(color=design.prim_colors_lst[1], width=1.5, dash="dash"),
        )
        fig.add_scatter(
            x=n,
            y=density,
            error_y=dict(type="data", array=density_sd, visible=True),
            mode="markers",
            name=f"minima ({strategy})",
            yaxis="y2",
            marker=dict(
                color=design.prim_colors_lst[0],
                size=design.marker_size,
                symbol=symbols[strategy],
            ),
        )
        fig.add_scatter(
            x=n,
            y=mts * gamma,
            mode="lines",
            showlegend=False,
            yaxis="y2",
            line=dict(color=design.prim_colors_lst[0], width=1.5, dash="dash"),
        )

    fig.update_layout(
        title="Landscape Hardness over Spectrum Size",
        xaxis=dict(title="$|\\Omega|$", type="log"),
        yaxis=dict(
            title="basin width", type="log", color=design.prim_colors_lst[1]
        ),
        yaxis2=dict(
            title="minima per unit scaler",
            type="log",
            overlaying="y",
            side="right",
            color=design.prim_colors_lst[0],
        ),
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
        margin=dict(b=160),
    )

    return fig


def landscape_over_circuits(df: pd.DataFrame):
    """
    Plot the basin width of the fixed-parameter loss for each ansatz, over the
    runs of the most frequent encoding, normalised by the Dirichlet prediction.

    The concentrated loss depends on the comb and the target only, so its
    curves are bitwise identical across ansätze and cannot answer this
    question. The fixed-parameter loss is the slice that does see the trainable
    unitary, through the coefficients, and the basin is the feature the
    Dirichlet geometry pins down: $2/(mts \\cdot \\gamma)$, set by the
    encoding generator alone. Normalising by it makes the gates of one run
    comparable, so all of them are pooled.

    The shaded band is the global mean plus and minus the spread of the seeds
    within one ansatz, i.e. the noise floor of the measurement. Ansatz means
    that sit inside it are not distinguishable from re-drawing the seed with
    the same ansatz, which is the sense in which the geometry does not depend
    on the circuit.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``generate_df``.

    Returns:
        go.Figure: A figure showing the normalised basin width per ansatz.
    """
    rows = _landscape_rows(df)

    # the gates of one run are made comparable by the normalisation, but a
    # different encoding changes which generators exist at all, so the figure
    # sticks to the configuration most runs share
    configs = [(row["model.encoding_strategy"], row["model.n_qubits"]) for row in rows]
    config = max(set(configs), key=configs.count)

    by_ansatz, by_seed = {}, {}
    for row, entry in zip(rows, configs):
        if entry != config:
            continue
        mts = row["landscape.mts"]
        for key in _landscape_gates(row):
            generator = row[f"landscape.generator.{key}"]
            width, _ = _basin_and_minima(
                np.array(row[f"landscape.eta.{key}.values"]),
                np.array(row[f"landscape.fixed.{key}.values"]),
                row[f"landscape.target_eta.{key}"],
            )
            ratio = width / (2.0 / (mts * generator))
            by_ansatz.setdefault(row["ansatz"], []).append(ratio)
            by_seed.setdefault((row["ansatz"], generator), []).append(ratio)

    ansatzes = sort_ansatzes(by_ansatz)
    labels = [circuit_name_to_str(a) for a in ansatzes]
    means = [np.nanmean(by_ansatz[a]) for a in ansatzes]
    overall = np.nanmean([v for values in by_ansatz.values() for v in values])
    # spread of the seeds at a fixed ansatz and generator, i.e. what the same
    # circuit gives when only the draw changes
    noise = np.nanmean(
        [np.nanstd(v) for v in by_seed.values() if len(v) > 1 and not np.all(np.isnan(v))]
    )

    fig = go.Figure()
    fig.add_hrect(
        y0=overall - noise,
        y1=overall + noise,
        fillcolor=design.prim_colors_lst[1],
        opacity=0.15,
        line_width=0,
    )
    fig.add_hline(
        y=overall, line=dict(color=design.prim_colors_lst[1], width=1.5, dash="dash")
    )

    for it, ansatz in enumerate(ansatzes):
        fig.add_scatter(
            x=[labels[it]] * len(by_ansatz[ansatz]),
            y=by_ansatz[ansatz],
            mode="markers",
            showlegend=False,
            marker=dict(
                color=design.prim_colors_lst[0],
                size=design.marker_size * 0.4,
                opacity=0.45,
            ),
        )

    fig.add_scatter(
        x=labels,
        y=means,
        mode="markers",
        name="ansatz mean",
        marker=dict(
            color=design.prim_colors_lst[0],
            size=design.marker_size,
            symbol="diamond",
        ),
    )
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="all ansätze, seed spread",
        line=dict(color=design.prim_colors_lst[1], width=1.5, dash="dash"),
    )

    fig.update_layout(
        title=f"Basin Width over Ansatz ({config[0]}, {config[1]:.0f} qubits)",
        xaxis=dict(tickangle=-60),
        yaxis=dict(title="basin width / prediction", rangemode="tozero"),
        template=design.template,
        font=dict(size=design.font_size),
        legend=dict(
            orientation="h", yanchor="top", y=-0.45, xanchor="center", x=0.5
        ),
        margin=dict(b=240),
    )

    return fig


def _coeff_columns(df: pd.DataFrame, prefix: str = "coeff.mean.f"):
    """Return the coefficient columns of `df` with their frequencies.

    The frequency is encoded in the column name, so a spectrum sampled with
    $mts > 1$ yields non-integer entries here. Sorted by frequency so the
    columns can be plotted directly against it.

    Args:
        df (pd.DataFrame): DataFrame carrying the coefficient columns.
        prefix (str): Column name prefix to match.

    Returns:
        tuple[list[str], list[float]]: Column names and their frequencies.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]
    freqs = [float(c.split(prefix)[1]) for c in cols]
    order = np.argsort(freqs)

    return [cols[i] for i in order], [freqs[i] for i in order]


def spectrum_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Plot the coefficient magnitude over frequency, one trace per pulse
    parameter variance, for each ansatz.

    The frequency axis is oversampled (see the `mts` parameter of the
    spectrum study), so the bins between the integers are the ones that can
    only be populated once the encoding gates acquire a frequency shift.
    An undistorted model puts all of its mass on the integer bins, so any
    growth in between is the effect under test.

    Args:
        df (pd.DataFrame): DataFrame with coeff.mean.f* columns,
            ``ansatz`` and ``pulse_params_variance``.
        max_distortion: Upper bound on pulse_params_variance to include.
        show_error: Whether to display error bars (std over seeds).

    Returns:
        List[go.Figure]: One figure per ansatz.
    """
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]
    coeff_cols, freqs = _coeff_columns(filtered_df)

    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    variances = sorted(filtered_df["pulse_params_variance"].unique())
    colors = plotly.colors.sample_colorscale(design.seq_colors, len(variances))

    figures = []
    for ansatz in ansatzes[:10]:
        ansatz_df = filtered_df[filtered_df["ansatz"] == ansatz]

        fig = go.Figure()
        for variance, color in zip(variances, colors):
            subset = ansatz_df[ansatz_df["pulse_params_variance"] == variance]
            if subset.empty:
                continue

            # average over seeds, keeping the frequency axis
            means = subset[coeff_cols].mean()
            stds = subset[coeff_cols].std()

            fig.add_scatter(
                x=freqs,
                y=means.values,
                error_y=dict(type="data", array=stds.values, visible=show_error),
                mode="lines+markers",
                name=f"{variance}",
                line=dict(color=color, width=design.marker_line_width),
                marker=dict(size=design.marker_size / 3),
            )

        fig.update_yaxes(type="log")
        # only the integer frequencies carry a label, the bins in between
        # are the ones the distortion fills in
        fig.update_xaxes(
            tickmode="array",
            tickvals=[f for f in freqs if f == int(f)],
        )

        fig.update_layout(
            title=f"Spectrum over PP Var. - {circuit_name_to_str(ansatz)}",
            xaxis_title="Frequency",
            yaxis_title="Mean |c|",
            template=design.template,
            font=dict(size=design.font_size),
            legend=design.horizontal_legend(),
        )

        figures.append(fig)

    return figures


def offgrid_mass_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Plot the share of coefficient magnitude sitting on non-integer
    frequencies over the pulse parameter variance.

    This condenses the spectrum into the single quantity the hypothesis is
    about: an undistorted model is supported on the integer frequencies
    alone, so a rising off-grid share means the encoding gates shifted the
    frequencies away from that grid.

    Args:
        df (pd.DataFrame): DataFrame with coeff.mean.f* columns,
            ``ansatz`` and ``pulse_params_variance``.
        max_distortion: Upper bound on pulse_params_variance to include.
        show_error: Whether to display error bars (std over seeds).
    """
    fig = go.Figure()

    filtered_df = df[df["pulse_params_variance"] <= max_distortion].copy()
    coeff_cols, freqs = _coeff_columns(filtered_df)
    off_cols = [c for c, f in zip(coeff_cols, freqs) if f != int(f)]

    filtered_df["offgrid_mass"] = (
        filtered_df[off_cols].sum(axis=1) / filtered_df[coeff_cols].sum(axis=1)
    )

    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    for ansatz in ansatzes[:10]:
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the off-grid mass over different seeds for a given distortion
        grouped_df = circuit_df.groupby("pulse_params_variance").offgrid_mass
        mean_mass = grouped_df.mean()
        std_mass = grouped_df.std()

        fig.add_scatter(
            x=mean_mass.index,
            y=mean_mass.values,
            error_y=dict(type="data", array=std_mass.values, visible=show_error),
            mode="lines",
            name=f"{circuit_name_to_str(ansatz)}",
            line=dict(color=next(color_it), width=design.marker_line_width),
        )

    fig.update_layout(
        title="Off-Grid Mass over PP Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Off-Grid Mass",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    return fig


def coeff_mean_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Given a dataframe with fccs for different distortions,
    plot the fcc over the distortions

    Args:
        df (pd.DataFrame): _description_
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    coeff_cols = [col for col in df.columns if col.startswith("coeff.var.f")]
    freq_indices = sorted([float(col.split("coeff.var.f")[1]) for col in coeff_cols])

    # Filter rows where pulse_params_variance is less than max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    variances = sorted(filtered_df["pulse_params_variance"].unique())

    symbol_it = iter(design.symbols_lst)
    # Create a trace for each circuit type
    for ansatz in ansatzes:

        color_it = iter(
            plotly.colors.sample_colorscale(design.seq_colors, len(variances))
        )
        symbol = next(symbol_it)
        for variance in variances:
            # Filter data for this circuit type
            circuit_distortion_df = filtered_df[
                (filtered_df["ansatz"] == ansatz)
                & (filtered_df["pulse_params_variance"] == variance)
            ]

            means = (
                circuit_distortion_df[[f"coeff.mean.f{idx}" for idx in freq_indices]]
                .mean()
                .values
            )
            stds = (
                circuit_distortion_df[[f"coeff.mean.f{idx}" for idx in freq_indices]]
                .std()
                .values
            )

            fig.add_scatter(
                x=freq_indices,
                y=means,
                mode="lines+markers",
                showlegend=False,
                marker=dict(
                    size=design.marker_size,
                    line=dict(width=design.marker_line_width),
                    symbol=symbol,
                ),
                line=dict(color=next(color_it)),
            )

    symbol_it = iter(design.symbols_lst)
    for ansatz in ansatzes:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"{circuit_name_to_str(ansatz)}",
            legendgroup=f"circuit",
            showlegend=True,
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
                symbol=next(symbol_it),
                color="gray",
            ),
        )

    color_it = iter(plotly.colors.sample_colorscale(design.seq_colors, len(variances)))
    for it, variance in enumerate(variances):
        color = next(color_it)

        if it > 0 and it < len(variances) - 1:
            continue
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"σ²={variance}",
            legendgroup=f"variance",
            showlegend=True,
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
                symbol="circle",
                color=color,
            ),
        )

    fig.update_layout(
        title="Coeff. Mean over Pulse Parameter Var.",
        xaxis_title="Frequency",
        yaxis_title="Coefficient Mean",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    fig.update_yaxes(type="log")

    return fig


def frequency_histogram_by_distortion(
    df: pd.DataFrame, max_distortion, threshold, show_error
):
    """
    Plot the number of active frequencies (|coeff| > threshold) per circuit,
    colored by distortion level.  Each (circuit, variance) combination is
    shown as a dot whose color encodes the pulse-parameter variance,
    using the same sequential colorscale as ``coeff_var_over_distortion``.

    Args:
        df (pd.DataFrame): DataFrame with coeff.var.f* columns,
            ``ansatz`` and ``pulse_params_variance``.
        max_distortion: Upper bound on pulse_params_variance to include.
        show_error: Whether to display error bars (std over seeds).
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    coeff_cols = [col for col in df.columns if col.startswith("coeff.var.f")]
    freq_indices = sorted([float(col.split("coeff.var.f")[1]) for col in coeff_cols])
    var_cols = [f"coeff.var.f{idx}" for idx in freq_indices]

    # Filter rows where pulse_params_variance is at most max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types sorted by ratio of pulse params to standard params
    ansatzes = sorted(
        filtered_df["ansatz"].unique(),
        key=lambda a: (
            filtered_df.loc[filtered_df["ansatz"] == a, "model.n_pulse_params"].iloc[0]
            # / filtered_df.loc[filtered_df["ansatz"] == a, "model.n_gate_params"].iloc[0]
        ),
    )
    variances = sorted(filtered_df["pulse_params_variance"].unique())
    x_labels = [circuit_name_to_str(a) for a in ansatzes]

    # Build a normalized color value in [0, 1] for each variance level
    var_min = variances[0]
    var_max = variances[-1]

    colors = plotly.colors.sample_colorscale(design.seq_colors, len(variances))

    # Plot data traces (one per variance level, shared color across circuits)
    for variance, color in zip(reversed(variances), reversed(colors)):
        means = []
        stds = []
        for ansatz in ansatzes:
            subset = filtered_df[
                (filtered_df["ansatz"] == ansatz)
                & (filtered_df["pulse_params_variance"] == variance)
            ]
            # Per-seed: count frequencies whose var coefficient > threshold
            n_freqs_per_seed = (subset[var_cols].abs() > threshold).sum(axis=1)
            means.append(n_freqs_per_seed.mean())
            stds.append(n_freqs_per_seed.std())

        fig.add_scatter(
            x=x_labels,
            y=means,
            error_y=dict(type="data", array=stds, visible=show_error),
            mode="markers",
            showlegend=False,
            marker=dict(
                size=design.marker_size,
                color=color,
                line=dict(width=design.marker_line_width),
            ),
        )

    # Add an invisible scatter trace solely to render the colorbar
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="markers",
        showlegend=False,
        marker=dict(
            size=0,
            color=[var_min, var_max],
            colorscale=design.seq_colors,
            showscale=True,
            colorbar=dict(
                title=dict(text="σ²", side="right"),
                thickness=15,
                tickvals=[var_min, var_max],
                ticktext=[str(var_min), str(var_max)],
            ),
        ),
    )

    fig.update_layout(
        title="# of Frequencies over PP Var.",
        xaxis_title="Circuit",
        yaxis_title="# of Frequencies",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
        xaxis_tickangle=-90,
    )

    fig.update_yaxes(dtick=1)

    return fig


def coeff_var_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Given a dataframe with fccs for different distortions,
    plot the fcc over the distortions

    Args:
        df (pd.DataFrame): _description_
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    coeff_cols = [col for col in df.columns if col.startswith("coeff.var.f")]
    freq_indices = sorted([float(col.split("coeff.var.f")[1]) for col in coeff_cols])

    # Filter rows where pulse_params_variance is less than max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    variances = sorted(filtered_df["pulse_params_variance"].unique())
    COEFF_VAR_CUTOFF = 5e-9

    symbol_it = iter(design.symbols_lst)
    # Create a trace for each circuit type
    for ansatz in ansatzes:

        color_it = iter(
            plotly.colors.sample_colorscale(design.seq_colors, len(variances))
        )
        symbol = next(symbol_it)
        for variance in variances:
            # Filter data for this circuit type
            circuit_distortion_df = filtered_df[
                (filtered_df["ansatz"] == ansatz)
                & (filtered_df["pulse_params_variance"] == variance)
            ]

            means = (
                circuit_distortion_df[[f"coeff.var.f{idx}" for idx in freq_indices]]
                .mean()
                .values
            )
            stds = (
                circuit_distortion_df[[f"coeff.var.f{idx}" for idx in freq_indices]]
                .std()
                .values
            )

            # Clamp coefficient variance values below the cutoff
            means_clamped = np.clip(means, a_min=COEFF_VAR_CUTOFF, a_max=None)

            fig.add_scatter(
                x=freq_indices,
                y=means_clamped,
                mode="lines+markers",
                showlegend=False,
                marker=dict(
                    size=design.marker_size,
                    line=dict(width=design.marker_line_width),
                    symbol=symbol,
                ),
                line=dict(color=next(color_it)),
            )

    # Add a horizontal dashed line at the cutoff as a visual indicator
    fig.add_hline(
        y=COEFF_VAR_CUTOFF,
        line_dash="dash",
        line_color="gray",
        line_width=1.5,
        annotation_text=f"cutoff = {COEFF_VAR_CUTOFF:.0e}",
        annotation_position="bottom right",
        annotation_font_size=design.font_size - 4,
        annotation_font_color="gray",
    )

    symbol_it = iter(design.symbols_lst)
    for ansatz in ansatzes:
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"{circuit_name_to_str(ansatz)}",
            legendgroup=f"circuit",
            showlegend=True,
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
                symbol=next(symbol_it),
                color="gray",
            ),
        )

    color_it = iter(plotly.colors.sample_colorscale(design.seq_colors, len(variances)))
    for it, variance in enumerate(variances):
        color = next(color_it)

        if it > 0 and it < len(variances) - 1:
            continue
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=f"σ²={variance}",
            legendgroup=f"variance",
            showlegend=True,
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
                symbol="circle",
                color=color,
            ),
        )

    fig.update_layout(
        title="Coeff. Var. over Pulse Parameter Var.",
        xaxis_title="Frequency",
        yaxis_title="Coefficient Variance",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    fig.update_yaxes(type="log")

    return fig


def coeff_var_delta_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Plot the difference in coefficient variance between zero distortion
    (pulse_params_variance == 0) and maximal distortion per ansatz over
    the frequency index.

    Args:
        df (pd.DataFrame): DataFrame with coeff.var.f* columns.
        max_distortion: Upper bound used to determine the maximal distortion level.
        show_error: Whether to display error bars.
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    coeff_cols = [col for col in df.columns if col.startswith("coeff.var.f")]
    freq_indices = sorted([float(col.split("coeff.var.f")[1]) for col in coeff_cols])
    var_cols = [f"coeff.var.f{idx}" for idx in freq_indices]

    # Filter rows where pulse_params_variance is at most max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types and determine the maximal variance present
    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    variances = sorted(filtered_df["pulse_params_variance"].unique())
    max_var = max(variances)

    color_it = iter(design.prim_colors_lst)

    for ansatz in ansatzes[:10]:
        ansatz_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # Baseline: zero distortion (variance == 0)
        baseline_df = ansatz_df[ansatz_df["pulse_params_variance"] == 0]
        baseline_means = baseline_df[var_cols].mean().values

        # Maximal distortion
        max_dist_df = ansatz_df[ansatz_df["pulse_params_variance"] == max_var]
        max_dist_means = max_dist_df[var_cols].mean().values

        # Relative change: ratio of maximal distortion to zero distortion
        # A value > 1 means distortion increased the coeff variance,
        # a value < 1 means it decreased.
        # Guard against division by zero with a small epsilon.
        epsilon = 1e-30
        delta = max_dist_means / np.maximum(baseline_means, epsilon)

        # Propagate uncertainty via error propagation for f = a/b:
        # σ_f/f = sqrt((σ_a/a)² + (σ_b/b)²)
        baseline_stds = baseline_df[var_cols].std().values
        max_dist_stds = max_dist_df[var_cols].std().values
        rel_err = np.sqrt(
            (np.nan_to_num(max_dist_stds) / np.maximum(max_dist_means, epsilon)) ** 2
            + (np.nan_to_num(baseline_stds) / np.maximum(baseline_means, epsilon)) ** 2
        )
        delta_stds = delta * rel_err

        color = next(color_it)

        fig.add_scatter(
            x=freq_indices,
            y=delta,
            error_y=dict(type="data", array=delta_stds, visible=show_error),
            mode="lines+markers",
            name=f"{circuit_name_to_str(ansatz)}",
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
            ),
            line=dict(color=color),
        )

    # Add a reference line at ratio = 1 (no change)
    fig.add_hline(
        y=1.0,
        line_dash="dash",
        line_color="gray",
        line_width=1.5,
    )

    fig.update_layout(
        title=f"Coeff. Var. Ratio (σ²={max_var} / σ²=0)",
        xaxis_title="Frequency",
        yaxis_title="Coefficient Variance Ratio",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    fig.update_yaxes(type="log")

    return fig


def fcc_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Given a dataframe with fccs for different distortions,
    plot the fcc over the distortions

    Args:
        df (pd.DataFrame): _description_
    """
    fig = go.Figure()

    # Filter rows where pulse_params_variance is less than max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes[:10]:
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fcc over different seeds for a given distortion
        grouped_df = circuit_df.groupby("pulse_params_variance").fcc
        mean_fcc = grouped_df.mean()
        std_fcc = grouped_df.std()

        fig.add_scatter(
            x=mean_fcc.index,
            y=mean_fcc.values,
            error_y=dict(type="data", array=std_fcc.values, visible=show_error),
            mode="lines",
            name=f"{circuit_name_to_str(ansatz)}",
            line=dict(color=next(color_it), width=design.marker_line_width),
        )

    fig.update_yaxes(type="log")

    fig.update_layout(
        title="FCC over PP Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="FCC",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    return fig


def fidelity_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Given a dataframe with fccs for different distortions,
    plot the fcc over the distortions

    Args:
        df (pd.DataFrame): _description_
    """
    fig = go.Figure()

    # Filter rows where pulse_params_variance is less than max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)
    
    # Create a trace for each circuit type
    for ansatz in ansatzes[:10]:  # TODO: just ignore some circuits which are redundant
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fidelity over different seeds for a given distortion
        grouped_df = circuit_df.groupby("pulse_params_variance")["fidelity"]
        mean = 1-grouped_df.mean() #infidelity
        std = grouped_df.std()

        fig.add_scatter(
            x=mean.index,
            y=mean.values,
            error_y=dict(type="data", array=std.values, visible=show_error),
            mode="lines",
            name=f"{circuit_name_to_str(ansatz)}",
            line=dict(color=next(color_it), width=design.marker_line_width),
        )
    
    fig.update_layout(
        title="Infidelity over PP Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Infidelity",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    fig.update_yaxes(type="log")

    return fig


def trace_distance_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Given a dataframe with fccs for different distortions,
    plot the fcc over the distortions

    Args:
        df (pd.DataFrame): _description_
    """
    fig = go.Figure()

    # Filter rows where pulse_params_variance is less than max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes[:10]:  # TODO: just ignore some circuits which are redundant
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fidelity over different seeds for a given distortion
        grouped_df = circuit_df.groupby("pulse_params_variance")["trace-distance"]
        mean = grouped_df.mean()
        std = grouped_df.std()

        fig.add_scatter(
            x=mean.index,
            y=mean.values,
            error_y=dict(type="data", array=std.values, visible=show_error),
            mode="lines",
            name=f"{circuit_name_to_str(ansatz)}",
            line=dict(color=next(color_it), width=design.marker_line_width),
        )

    fig.update_layout(
        title="Trace Distance over Pulse Parameter Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Trace Distance",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    fig.update_yaxes(type="log")

    return fig


def expressibility_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    fig = go.Figure()

    # Filter rows where pulse_params_variance is less than max_distortion
    filtered_df = df[df["pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes[: len(design.prim_colors_lst)]:
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fidelity over different seeds for a given distortion
        grouped_df = circuit_df.groupby("pulse_params_variance")["expressibility"]
        mean = grouped_df.mean()
        std = grouped_df.std()

        fig.add_scatter(
            x=mean.index,
            y=mean.values,
            error_y=dict(type="data", array=std.values, visible=show_error),
            mode="lines",
            name=f"{circuit_name_to_str(ansatz)}",
            line=dict(color=next(color_it), width=design.marker_line_width),
        )

    fig.update_layout(
        title="Expr. over PP Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Expressibility",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    fig.update_yaxes(type="log")

    return fig


def pulse_param_mse_comparison(
    df: pd.DataFrame,
    show_error: bool = True,
):
    """
    Compare the train MSE across circuits for the unitary and pulse gate modes.
    Produces a grouped bar chart with circuits on the x-axis and two bars per
    circuit (one for each gate mode), including error bars over seeds.

    Args:
        df (pd.DataFrame): DataFrame with columns "ansatz", "gate_mode",
            "train_mse", "run_id", and "data.seed".
        show_error (bool): Whether to display error bars. Defaults to True.

    Returns:
        go.Figure: The plotly figure.
    """
    fig = go.Figure()

    # Sort ansatzes by ratio of pulse params to standard params
    ansatzes = sorted(
        df["ansatz"].unique(),
        key=lambda a: (
            df.loc[df["ansatz"] == a, "model.n_pulse_params"].iloc[0]
            # / df.loc[df["ansatz"] == a, "model.n_gate_params"].iloc[0]
        ),
    )
    x_labels = [circuit_name_to_str(a) for a in ansatzes]

    color_it = iter(design.prim_colors_lst)
    cases = [
        ("unitary", False, "Gate"),
        ("enc_pulse", False, "+ Pulse"),
        # ("ansatz_pulse", False, "+ Pulse"),
        ("unitary", True, "Decomposed"),
    ]
    for gate_mode, decompose_circuit, label in cases:
        color = next(color_it)

        means = []
        stds = []
        for ansatz in ansatzes:
            subset = df[df["ansatz"] == ansatz]
            subset = subset[subset["gate_mode"] == gate_mode]
            subset = subset[subset["decompose_circuit"] == decompose_circuit]

            means.append(subset['train_mse'].mean())
            stds.append(subset['train_mse'].std())

        fig.add_bar(
            x=x_labels,
            y=means,
            error_y=dict(type="data", array=stds, visible=show_error),
            name=label,
            marker=dict(color=color),
        )

    fig.update_layout(
        title=f"MSE: Gate vs. Gate + Pulse",
        xaxis_title="Circuit",
        yaxis_title="MSE",
        barmode="group",
        template=design.template,
        font=dict(size=design.font_size),
        legend=design.horizontal_legend(),
    )

    return fig


def pulse_mean_and_variance_over_step(
    df: pd.DataFrame, show_error: bool = True
):
    """
    Visualize how pulse_scaler_mean and pulse_scaler_std evolve over training
    steps.  For each ansatz the per-step metric data is read from the
    list-valued columns on ``df`` itself, averaged over seeds, and plotted
    with optional error bars.

    Args:
        df (pd.DataFrame): DataFrame with columns "run_id", "ansatz",
            "gate_mode" and the list-valued step/value columns produced
            by ``generate_df``.  Only rows with a pulse-level gate mode are
            considered.
        show_error (bool): Whether to display error bars (std over seeds).

    Returns:
        tuple[go.Figure, go.Figure]: Two figures – one for pulse_scaler_mean
            and one for pulse_scaler_std over training steps.
    """
    filtered_df = df[df["gate_mode"].isin(["ansatz_pulse", "all_pulse", "enc_pulse"])]

    ansatzes = sort_ansatzes(filtered_df["ansatz"].unique())

    figures = []
    for metric_name, y_label, title in [
        ("pulse_scaler_mean", "Pulse Scaler Mean", "Pulse Scaler Mean over Step"),
        ("pulse_scaler_std", "Pulse Scaler Std", "Pulse Scaler Std over Step"),
    ]:
        fig = go.Figure()
        color_it = iter(design.prim_colors_lst)

        for ansatz in ansatzes[:10]:
            ansatz_df = filtered_df[filtered_df["ansatz"] == ansatz]
            hist_df = _collect_metric_history(ansatz_df, metric_name)
            if hist_df.empty:
                # enc_pulse runs log e.g. "enc_pulse_scaler_mean"
                hist_df = _collect_metric_history(ansatz_df, f"enc_{metric_name}")

            if hist_df.empty:
                continue

            steps = hist_df.index.values
            mean_vals = hist_df.mean(axis=1).values
            std_vals = hist_df.std(axis=1).values

            color = next(color_it)

            fig.add_scatter(
                x=steps,
                y=mean_vals,
                mode="lines",
                name=circuit_name_to_str(ansatz),
                line=dict(color=color, width=1.5),
                legendgroup=ansatz,
            )

            if show_error:
                # Add shaded area for standard deviation
                fig.add_scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate(
                        [mean_vals + std_vals, (mean_vals - std_vals)[::-1]]
                    ),
                    fill="toself",
                    fillcolor=(
                        color.replace("rgb", "rgba").replace(")", ", 0.2)")
                        if "rgb" in color
                        else color
                    ),
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    legendgroup=ansatz,
                    hoverinfo="skip",
                )

        fig.update_layout(
            title=title,
            xaxis_title="Step",
            yaxis_title=y_label,
            template=design.template,
            font=dict(size=design.font_size),
            legend=design.horizontal_legend(),
        )

        figures.append(fig)

    return figures


def loss_over_step(
    df: pd.DataFrame, show_error: bool = True
):
    """
    Visualize how the training loss evolves over training steps for each ansatz.
    For each ansatz the per-step metric data is read from the list-valued
    columns on ``df`` itself, averaged over seeds, and plotted with optional
    error bars.

    Args:
        df (pd.DataFrame): DataFrame with columns "run_id", "ansatz",
            "gate_mode" and the list-valued step/value columns produced
            by ``generate_df``.
        show_error (bool): Whether to display error bars (std over seeds).

    Returns:
        go.Figure: A figure showing loss over training steps.
    """
    ansatzes = sort_ansatzes(df["ansatz"].unique())

    fig = go.Figure()
    color_it = iter(design.prim_colors_lst)
    ansatz_colors = {}

    for ansatz in ansatzes[:10]:
        color = next(color_it)
        ansatz_colors[ansatz] = color

        for gate_mode, dash_style in [
            ("enc_pulse", "solid"),
            ("unitary", "dash"),
        ]:
            subset = df[(df["ansatz"] == ansatz) & (df["gate_mode"] == gate_mode)]
            if subset.empty:
                continue

            # Try common loss metric names; both live as list-valued columns
            # on ``subset`` now.
            hist_df = _collect_metric_history(subset, 'train_mse')
            if hist_df.empty:
                hist_df = _collect_metric_history(subset, "loss")
            if hist_df.empty:
                continue
            
            steps = hist_df.index.values
            mean_vals = hist_df.mean(axis=1).values
            std_vals = hist_df.std(axis=1).values

            legend_group = f"{ansatz}_{gate_mode}"

            fig.add_scatter(
                x=steps,
                y=mean_vals,
                mode="lines",
                showlegend=False,
                line=dict(color=color, width=1.5, dash=dash_style),
                legendgroup=legend_group,
            )

            if show_error:
                # Add shaded area for standard deviation
                fig.add_scatter(
                    x=np.concatenate([steps, steps[::-1]]),
                    y=np.concatenate(
                        [mean_vals + std_vals, (mean_vals - std_vals)[::-1]]
                    ),
                    fill="toself",
                    fillcolor=(
                        color.replace("rgb", "rgba").replace(")", ", 0.2)")
                        if "rgb" in color
                        else color
                    ),
                    line=dict(color="rgba(0,0,0,0)"),
                    showlegend=False,
                    legendgroup=legend_group,
                    hoverinfo="skip",
                )

    # Add legend entries for each ansatz (colored, solid)
    for ansatz, color in ansatz_colors.items():
        fig.add_scatter(
            x=[None],
            y=[None],
            mode="lines",
            name=circuit_name_to_str(ansatz),
            line=dict(color=color, width=1.5),
            showlegend=True,
            legendgroup="circuits",
        )

    # Add legend entries for line style (grey)
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="+ Pulse (solid)",
        line=dict(color="gray", width=1.5, dash="solid"),
        showlegend=True,
        legendgroup="styles",
    )
    fig.add_scatter(
        x=[None],
        y=[None],
        mode="lines",
        name="Gate (dashed)",
        line=dict(color="gray", width=1.5, dash="dash"),
        showlegend=True,
        legendgroup="styles",
    )

    fig.update_layout(
        title="Loss over Step",
        xaxis_title="Step",
        yaxis_title="Loss",
        template=design.template,
        font=dict(size=design.font_size),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.25,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(b=120),
    )

    fig.update_yaxes(type="log")

    return fig
