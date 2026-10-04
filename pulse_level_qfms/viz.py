"""The studies' figures, drawn from the table of their runs.

Plain plotting over a DataFrame with the columns :mod:`pulse_level_qfms.table`
writes: nothing here talks to the engine or knows which study it draws. Each
study's ``dev/<study>/figures.py`` picks the figures it needs and hands them
to :func:`save`.
"""

import re
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelmax, argrelmin


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
    legend_color = "#002D4C"
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


def save(figures: Dict[str, go.Figure], folder: Path) -> None:
    """Write each figure as ``<folder>/<name>.pdf``."""
    folder.mkdir(parents=True, exist_ok=True)
    for name, fig in figures.items():
        path = folder / f"{name}.pdf"
        print(f"Saving figure to {path}")
        fig.write_image(path)


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

    Each panel sweeps one scaler while the others stay at their target values.
    For generator $\\gamma$, loss oscillates with period
    $1/(mts \\cdot \\gamma)$; panels share the scaler axis but use separate
    loss axes.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``table``.
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
    """Measure basin width and minima density along a scaler sweep.

    The basin spans the maxima flanking the minimum at ``target``. Density is
    the number of minima between 1 and ``target`` divided by that distance.

    Args:
        grid (np.ndarray): The scaler grid.
        values (np.ndarray): Loss over the grid.
        target (float): The aligned scaler.

    Returns:
        tuple[float, float]: Basin width and local minima per unit scaler, both
        NaN if the curve is flat.
    """
    # an ansatz whose coefficients vanish on the components the swept gate
    # drives leaves the loss independent of that scaler. What is left is
    # rounding noise, whose extrema carry no geometry, so such a slice is
    # dropped rather than measured.
    span = np.nanmax(values) - np.nanmin(values)
    if span <= 1e-9 * abs(np.nanmean(values)):
        return np.nan, np.nan

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
    Plot basin width and minima density against each gate's generator.

    The concentrated loss gives a reference basin width
    $2/(mts \\cdot \\gamma)$ and minima density $mts \\cdot \\gamma$ for
    generator $\\gamma$. Markers at one generator vary with the target draw;
    minima counts are coarse on short paths.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``table``.

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
    Plot basin width and minima density against model spectrum size.

    Use the largest encoding generator in each run and one trace per strategy.

    Use basin width and minima density for gates with the largest generator
    $\\gamma_{max}$, averaging runs of the same spectrum size. Dirichlet
    references are $2/(mts \\cdot \\gamma_{max})$ and
    $mts \\cdot \\gamma_{max}$. Error bars show variation across runs.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``table``, one row per run of the
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
    Compare normalized basin width and minima density across ansatzes.

    Use fixed-parameter loss from the most common encoding strategy.

    Use fixed-parameter loss so the measurements reflect each ansatz's
    coefficients. Normalize each gate by its Dirichlet references:
    $2/(mts \\cdot \\gamma)$ for basin width and $mts \\cdot \\gamma$ for
    minima density. Shaded bands show variation across seeds.

    Args:
        df (pd.DataFrame): DataFrame carrying the list-valued landscape
            columns produced by ``table``.

    Returns:
        go.Figure: A figure showing both quantities per ansatz.
    """
    rows = _landscape_rows(df)

    # the gates of one run are made comparable by the normalisation, but a
    # different encoding changes which generators exist at all, so the figure
    # sticks to the configuration most runs share
    configs = [(row["model.encoding_strategy"], row["model.n_qubits"]) for row in rows]
    config = max(set(configs), key=configs.count)

    quantities = ("basin width", "minima per unit scaler")
    # spelled out rather than called a prediction, so the divisor is visible
    axis_titles = ("$W \\;/\\; 2(mts \\cdot \\gamma)^{-1}$",
                   "$\\nu \\;/\\; mts \\cdot \\gamma$")
    colors = (design.prim_colors_lst[1], design.prim_colors_lst[0])
    by_ansatz = {q: {} for q in quantities}
    by_seed = {q: {} for q in quantities}

    for row, entry in zip(rows, configs):
        if entry != config:
            continue
        mts = row["landscape.mts"]
        for key in _landscape_gates(row):
            generator = row[f"landscape.generator.{key}"]
            width, density = _basin_and_minima(
                np.array(row[f"landscape.eta.{key}.values"]),
                np.array(row[f"landscape.fixed.{key}.values"]),
                row[f"landscape.target_eta.{key}"],
            )
            ratios = {
                "basin width": width / (2.0 / (mts * generator)),
                "minima per unit scaler": density / (mts * generator),
            }
            for quantity, ratio in ratios.items():
                if not np.isfinite(ratio):
                    continue
                by_ansatz[quantity].setdefault(row["ansatz"], []).append(ratio)
                by_seed[quantity].setdefault((row["ansatz"], generator), []).append(
                    ratio
                )

    ansatzes = sort_ansatzes(by_ansatz[quantities[0]])
    labels = [circuit_name_to_str(a) for a in ansatzes]

    fig = make_subplots(rows=len(quantities), cols=1, shared_xaxes=True,
                        vertical_spacing=0.07)

    for it, quantity in enumerate(quantities):
        values = by_ansatz[quantity]
        means = [np.nanmean(values[a]) for a in ansatzes]
        overall = np.nanmean([v for entries in values.values() for v in entries])
        # spread of the seeds at a fixed ansatz and generator, i.e. what the
        # same circuit gives when only the draw changes
        noise = np.nanmean(
            [
                np.nanstd(v)
                for v in by_seed[quantity].values()
                if len(v) > 1 and not np.all(np.isnan(v))
            ]
        )

        fig.add_hrect(
            y0=overall - noise,
            y1=overall + noise,
            fillcolor=colors[it],
            opacity=0.15,
            line_width=0,
            row=it + 1,
            col=1,
        )
        fig.add_hline(
            y=overall,
            line=dict(color=colors[it], width=1.5, dash="dash"),
            row=it + 1,
            col=1,
        )

        for jt, ansatz in enumerate(ansatzes):
            fig.add_scatter(
                x=[labels[jt]] * len(values[ansatz]),
                y=values[ansatz],
                mode="markers",
                showlegend=False,
                marker=dict(
                    color=colors[it], size=design.marker_size * 0.4, opacity=0.45
                ),
                row=it + 1,
                col=1,
            )

        fig.add_scatter(
            x=labels,
            y=means,
            mode="markers",
            name=quantity,
            marker=dict(
                color=colors[it], size=design.marker_size, symbol="diamond"
            ),
            row=it + 1,
            col=1,
        )
        fig.update_yaxes(
            title_text=axis_titles[it],
            rangemode="tozero",
            row=it + 1,
            col=1,
        )

    fig.update_xaxes(tickangle=-60, row=len(quantities), col=1)
    fig.update_layout(
        title=f"Landscape Geometry over Ansatz ({config[0]}, {config[1]:.0f} qubits)",
        template=design.template,
        font=dict(size=design.font_size),
        legend=dict(
            orientation="h", yanchor="top", y=-0.3, xanchor="center", x=0.5
        ),
        margin=dict(b=240),
        width=1000,
        height=800,
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


def spectrum_over_distortion(df: pd.DataFrame, show_error):
    """
    Plot the coefficient magnitude over frequency, one trace per pulse
    parameter variance, for each ansatz.

    Non-integer bins reveal frequency shifts from distorted encoding gates.

    Args:
        df (pd.DataFrame): DataFrame with coeff.mean.f* columns,
            ``ansatz`` and ``pulse_params_variance``.
        show_error: Whether to display error bars (std over seeds).

    Returns:
        Dict[str, go.Figure]: One figure per ansatz, keyed by it.
    """
    coeff_cols, freqs = _coeff_columns(df)

    ansatzes = sort_ansatzes(df["ansatz"].unique())
    variances = sorted(df["pulse_params_variance"].unique())
    colors = plotly.colors.sample_colorscale(design.seq_colors, len(variances))

    figures = {}
    for ansatz in ansatzes[:10]:
        ansatz_df = df[df["ansatz"] == ansatz]

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

        figures[ansatz] = fig

    return figures


def offgrid_mass_over_distortion(df: pd.DataFrame, show_error):
    """
    Plot the share of coefficient magnitude sitting on non-integer
    frequencies over the pulse parameter variance.

    An undistorted model has no non-integer frequency mass.

    Args:
        df (pd.DataFrame): DataFrame with coeff.mean.f* columns,
            ``ansatz`` and ``pulse_params_variance``.
        show_error: Whether to display error bars (std over seeds).
    """
    fig = go.Figure()

    coeff_cols, freqs = _coeff_columns(df)
    off_cols = [c for c, f in zip(coeff_cols, freqs) if f != int(f)]

    df = df.assign(
        offgrid_mass=df[off_cols].sum(axis=1) / df[coeff_cols].sum(axis=1)
    )

    ansatzes = sort_ansatzes(df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    for ansatz in ansatzes[:10]:
        circuit_df = df[df["ansatz"] == ansatz]

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


def frequency_histogram_by_distortion(
    df: pd.DataFrame, threshold, show_error
):
    """
    Plot the number of active frequencies (|coeff| > threshold) per circuit,
    colored by distortion level.  Each (circuit, variance) combination is
    shown as a dot whose color encodes the pulse-parameter variance,
    using the same sequential colorscale as ``coeff_var_over_distortion``.

    Args:
        df (pd.DataFrame): DataFrame with coeff.var.f* columns,
            ``ansatz`` and ``pulse_params_variance``.
        show_error: Whether to display error bars (std over seeds).
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    coeff_cols = [col for col in df.columns if col.startswith("coeff.var.f")]
    freq_indices = sorted([float(col.split("coeff.var.f")[1]) for col in coeff_cols])
    var_cols = [f"coeff.var.f{idx}" for idx in freq_indices]


    # Get unique circuit types sorted by ratio of pulse params to standard params
    ansatzes = sorted(
        df["ansatz"].unique(),
        key=lambda a: (
            df.loc[df["ansatz"] == a, "model.n_pulse_params"].iloc[0]
            # / df.loc[df["ansatz"] == a, "model.n_gate_params"].iloc[0]
        ),
    )
    variances = sorted(df["pulse_params_variance"].unique())
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
            subset = df[
                (df["ansatz"] == ansatz)
                & (df["pulse_params_variance"] == variance)
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


def coeff_var_delta_over_distortion(df: pd.DataFrame, show_error):
    """
    Plot the difference in coefficient variance between zero distortion
    (pulse_params_variance == 0) and maximal distortion per ansatz over
    the frequency index.

    Args:
        df (pd.DataFrame): DataFrame with coeff.var.f* columns.
        show_error: Whether to display error bars.
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    coeff_cols = [col for col in df.columns if col.startswith("coeff.var.f")]
    freq_indices = sorted([float(col.split("coeff.var.f")[1]) for col in coeff_cols])
    var_cols = [f"coeff.var.f{idx}" for idx in freq_indices]


    # Get unique circuit types and determine the maximal variance present
    ansatzes = sort_ansatzes(df["ansatz"].unique())
    variances = sorted(df["pulse_params_variance"].unique())
    max_var = max(variances)

    color_it = iter(design.prim_colors_lst)

    for ansatz in ansatzes[:10]:
        ansatz_df = df[df["ansatz"] == ansatz]

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


def fcc_over_distortion(df: pd.DataFrame, show_error):
    """Plot mean Fourier coefficient concentration against pulse variance.

    Args:
        df: Runs with ``ansatz``, ``pulse_params_variance``, and ``fcc``.
        show_error: Show standard deviation across seeds.
    """
    fig = go.Figure()


    # Get unique circuit types
    ansatzes = sort_ansatzes(df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes[:10]:
        # Filter data for this circuit type
        circuit_df = df[df["ansatz"] == ansatz]

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


def fidelity_over_distortion(df: pd.DataFrame, show_error):
    """Plot mean infidelity against pulse variance for each ansatz.

    Args:
        df: Runs with ``ansatz``, ``pulse_params_variance``, and ``fidelity``.
        show_error: Show standard deviation across seeds.
    """
    fig = go.Figure()


    # Get unique circuit types
    ansatzes = sort_ansatzes(df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes[:10]:  # TODO: just ignore some circuits which are redundant
        # Filter data for this circuit type
        circuit_df = df[df["ansatz"] == ansatz]

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


def trace_distance_over_distortion(df: pd.DataFrame, show_error):
    """Plot mean trace distance against pulse variance for each ansatz.

    Args:
        df: Runs with ``ansatz``, ``pulse_params_variance``, and
            ``trace-distance``.
        show_error: Show standard deviation across seeds.
    """
    fig = go.Figure()


    # Get unique circuit types
    ansatzes = sort_ansatzes(df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes[:10]:  # TODO: just ignore some circuits which are redundant
        # Filter data for this circuit type
        circuit_df = df[df["ansatz"] == ansatz]

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


def expressibility_over_distortion(df: pd.DataFrame, show_error):
    fig = go.Figure()


    # Get unique circuit types
    ansatzes = sort_ansatzes(df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes[: len(design.prim_colors_lst)]:
        # Filter data for this circuit type
        circuit_df = df[df["ansatz"] == ansatz]

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
    """Compare training MSE for gate, pulse, and decomposed circuits.

    Args:
        df: Runs with ``ansatz``, ``train_pulse``, ``decompose_circuit``,
            ``train_mse``, and ``model.n_pulse_params``.
        show_error: Show standard deviation across seeds.

    Returns:
        Grouped bar chart by ansatz and training mode.
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
        (False, False, "Gate"),
        (True, False, "+ Pulse"),
        (False, True, "Decomposed"),
    ]
    for train_pulse, decompose_circuit, label in cases:
        color = next(color_it)

        means = []
        stds = []
        for ansatz in ansatzes:
            subset = df[df["ansatz"] == ansatz]
            subset = subset[subset["train_pulse"] == train_pulse]
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
        title="MSE: Gate vs. Gate + Pulse",
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
    """Plot pulse scaler mean and standard deviation over training steps.

    Uses runs with ``train_pulse=True`` and averages each curve over seeds.
    Falls back to encoding pulse scaler metrics when needed.

    Args:
        df: Runs with streamed metric step and value columns from ``table``.
        show_error: Show standard deviation across seeds.

    Returns:
        Figures for scaler mean and scaler standard deviation.
    """
    # Only consider runs that actually trained pulse parameters
    filtered_df = df[df["train_pulse"] == True]  # noqa: E712

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
    """Plot training MSE over steps by ansatz and pulse training mode.

    Curves are averaged over seeds from the step and value columns in ``table``.

    Args:
        df: Runs with ``ansatz``, ``train_pulse``, and streamed MSE columns.
        show_error: Show standard deviation across seeds.

    Returns:
        Training loss figure.
    """
    ansatzes = sort_ansatzes(df["ansatz"].unique())

    fig = go.Figure()
    color_it = iter(design.prim_colors_lst)
    ansatz_colors = {}

    for ansatz in ansatzes[:10]:
        color = next(color_it)
        ansatz_colors[ansatz] = color

        for train_pulse, dash_style in [
            (True, "solid"),
            (False, "dash"),
        ]:
            subset = df[(df["ansatz"] == ansatz) & (df["train_pulse"] == train_pulse)]
            if subset.empty:
                continue

            hist_df = _collect_metric_history(subset, 'train_mse')
            if hist_df.empty:
                continue

            steps = hist_df.index.values
            mean_vals = hist_df.mean(axis=1).values
            std_vals = hist_df.std(axis=1).values

            legend_group = f"{ansatz}_{train_pulse}"

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
