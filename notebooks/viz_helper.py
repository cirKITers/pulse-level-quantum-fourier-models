import os
import re
import plotly
import plotly.graph_objects as go
from typing import List
import numpy as np
import pandas as pd
import string


def _natural_sort_key(s: str):
    """Sort key that handles embedded numbers naturally.
    E.g. Circuit_3 < Circuit_8 < Circuit_13 instead of lexicographic order."""
    return [
        int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", s)
    ]


def sort_ansatzes(ansatzes):
    """Sort ansatz names using natural ordering."""
    return sorted(ansatzes, key=_natural_sort_key)


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
    prim_colors_lst = plotly.colors.qualitative.T10
    sec_colors_lst = plotly.colors.qualitative.Safe
    seq_colors = plotly.colors.sequential.dense_r


def circuit_name_to_str(circuit_name: str):
    if "Circuit" in circuit_name:
        circuit_name = circuit_name.replace("Circuit", "C")
    elif "Hardware_Efficient" in circuit_name:
        circuit_name = "HEA"

    circuit_name = circuit_name.replace("_", " ")

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


def viz_study_1(df, max_distortion, show_error):
    figures = []

    figures.append(fcc_over_distortion(df, max_distortion, show_error))
    figures.append(coeff_mean_over_distortion(df, max_distortion, show_error))
    figures.append(coeff_var_over_distortion(df, max_distortion, show_error))
    figures.append(coeff_var_delta_over_distortion(df, max_distortion, show_error))

    return figures


def viz_study_2(df, max_distortion, show_error):
    figures = []

    figures.append(fidelity_over_distortion(df, max_distortion, show_error))
    figures.append(trace_distance_over_distortion(df, max_distortion, show_error))

    return figures


def viz_study_4(df, show_error):
    figures = []

    figures.append(pulse_param_mse_comparison(df, show_error))

    return figures


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
            circuit_distortion_df = filtered_df[filtered_df["ansatz"] == ansatz][
                filtered_df["pulse_params_variance"] == variance
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
        legend_indentation=-12,
        legend_tracegroupgap=3,
    )

    fig.update_yaxes(type="log")

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
            circuit_distortion_df = filtered_df[filtered_df["ansatz"] == ansatz][
                filtered_df["pulse_params_variance"] == variance
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
        legend_indentation=-12,
        legend_tracegroupgap=3,
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

    for ansatz in ansatzes:
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
    for ansatz in ansatzes:
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
            mode="lines+markers",
            name=f"{circuit_name_to_str(ansatz)}",
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
            ),
            line=dict(color=next(color_it)),
        )

    fig.update_layout(
        title="FCC over PP Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="FCC",
        template=design.template,
        font=dict(size=design.font_size),
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
    for ansatz in ansatzes:
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fidelity over different seeds for a given distortion
        grouped_df = circuit_df.groupby("pulse_params_variance")["fidelity"]
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
        title="Fidelity over PP Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Fidelity",
        template=design.template,
        font=dict(size=design.font_size),
    )

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
    for ansatz in ansatzes:
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
    )

    return fig


def pulse_param_mse_comparison(df: pd.DataFrame, show_error: bool = True):
    """
    Compare the final train MSE across circuits for train_pulse=True vs False.
    Produces a grouped bar chart with circuits on the x-axis and two bars per
    circuit (one for each train_pulse setting), including error bars over seeds.

    Args:
        df (pd.DataFrame): DataFrame with columns "ansatz", "train_pulse",
            "train_mse", and "data.seed".
        show_error (bool): Whether to display error bars. Defaults to True.

    Returns:
        go.Figure: The plotly figure.
    """
    fig = go.Figure()

    ansatzes = sort_ansatzes(df["ansatz"].unique())
    x_labels = [circuit_name_to_str(a) for a in ansatzes]

    color_it = iter(design.prim_colors_lst)
    for train_pulse, label in [(False, "Unitary only"), (True, "Unitary + Pulse")]:
        color = next(color_it)

        means = []
        stds = []
        for ansatz in ansatzes:
            subset = df[(df["ansatz"] == ansatz) & (df["train_pulse"] == train_pulse)]
            means.append(subset["train_mse"].mean())
            stds.append(subset["train_mse"].std())

        fig.add_bar(
            x=x_labels,
            y=means,
            error_y=dict(type="data", array=stds, visible=show_error),
            name=label,
            marker=dict(color=color),
        )

    fig.update_layout(
        title="Train MSE: Unitary vs. Unitary + Pulse",
        xaxis_title="Circuit",
        yaxis_title="Train MSE",
        barmode="group",
        template=design.template,
        font=dict(size=design.font_size),
    )

    return fig
