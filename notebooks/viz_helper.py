import os
import plotly
import plotly.graph_objects as go
from typing import List
import pandas as pd
from copy import deepcopy
import string

from data_helper import generate_hash


class design:
    template = "plotly_white"
    font_size = 22
    marker_size = 14
    marker_line_width = 1
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
    ]
    prim_colors_lst = plotly.colors.qualitative.Dark2
    sec_colors_lst = plotly.colors.qualitative.Pastel2
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
    figures.append(coeff_var_over_distortion(df, max_distortion, show_error))

    return figures


def viz_study_2(df, max_distortion, show_error):
    figures = []

    figures.append(fidelity_over_distortion(df, max_distortion, show_error))
    figures.append(trace_distance_over_distortion(df, max_distortion, show_error))

    return figures


def _coeff_var_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Given a dataframe with variances for different frequencies
    and different distortions, plot the variances over the frequencies
    and use different traces for the distortions. Different circuit types
    are distinguished by marker shapes.

    Args:
        df (pd.DataFrame): DataFrame containing coefficient variances
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    coeff_cols = [col for col in df.columns if col.startswith("coeff.var.f")]
    freq_indices = sorted([float(col.split("coeff.var.f")[1]) for col in coeff_cols])

    # Filter by max distortion
    filtered_df = df[df["fcc.pulse_params_variance"] <= max_distortion].copy()

    # Get unique circuit types and distortions
    circuit_types = (
        sorted(filtered_df["ansatz"].unique())
        if "ansatz" in filtered_df.columns
        else [None]
    )
    distortions = sorted(filtered_df["fcc.pulse_params_variance"].unique())[
        :7
    ]  # Limit to 7

    # Create color mapping for distortions and symbol mapping for circuit types
    distortion_color_map = {
        dist: color for dist, color in zip(distortions, design.prim_colors_lst)
    }
    circuit_symbol_map = {
        ct: symbol for ct, symbol in zip(circuit_types, design.symbols_lst)
    }

    # Track which legend entries we've added
    distortion_legend_added = set()
    circuit_legend_added = set()

    # Create a trace for each distortion level and circuit type combination
    for distortion in distortions:
        distortion = float(distortion)
        color = distortion_color_map[distortion]

        # Group by circuit type for this distortion
        distortion_df = filtered_df[
            filtered_df["fcc.pulse_params_variance"] == distortion
        ]

        if "ansatz" in distortion_df.columns:
            for circuit_type in circuit_types:
                circuit_distortion_df = distortion_df[
                    distortion_df["ansatz"] == circuit_type
                ]

                # Calculate mean and std for each frequency using pandas
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

                # Determine if this should be shown in legend
                show_in_legend = (
                    distortion not in distortion_legend_added
                    or circuit_type not in circuit_legend_added
                )

                # Add trace with error bars
                fig.add_scatter(
                    x=freq_indices,
                    y=means,
                    error_y=dict(type="data", array=stds, visible=show_error),
                    mode="lines+markers",
                    name=(
                        f"σ²={distortion}"
                        if distortion not in distortion_legend_added
                        else ""
                    ),
                    legendgroup=f"distortion_{distortion}",
                    showlegend=distortion not in distortion_legend_added,
                    marker=dict(
                        size=design.marker_size,
                        line=dict(width=design.marker_line_width),
                        symbol=circuit_symbol_map[circuit_type],
                    ),
                    line=dict(color=color),
                )

                distortion_legend_added.add(distortion)

            # Add dummy traces for circuit type legend (shapes only)
            for circuit_type in circuit_types:
                if circuit_type not in circuit_legend_added:
                    fig.add_scatter(
                        x=[None],
                        y=[None],
                        mode="markers",
                        name=f"{circuit_name_to_str(circuit_type)}",
                        legendgroup=f"circuit_{circuit_type}",
                        showlegend=True,
                        marker=dict(
                            size=design.marker_size,
                            line=dict(width=design.marker_line_width),
                            symbol=circuit_symbol_map[circuit_type],
                            color="gray",
                        ),
                    )
                    circuit_legend_added.add(circuit_type)
        else:
            # Fallback if no circuit type column exists
            means = (
                distortion_df[[f"coeff.var.f{idx}" for idx in freq_indices]]
                .mean()
                .values
            )
            stds = (
                distortion_df[[f"coeff.var.f{idx}" for idx in freq_indices]]
                .std()
                .values
            )

            fig.add_scatter(
                x=freq_indices,
                y=means,
                error_y=dict(type="data", array=stds, visible=show_error),
                mode="lines+markers",
                name=f"σ²={distortion}",
                marker=dict(
                    size=design.marker_size,
                    line=dict(width=design.marker_line_width),
                ),
                line=dict(color=color),
            )

    fig.update_layout(
        title="Coefficient Variance over Frequency Indices",
        xaxis_title="Frequency Index",
        yaxis_title="Coefficient Variance",
        template=design.template,
        font=dict(size=design.font_size),
    )

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
    filtered_df = df[df["fcc.pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sorted(filtered_df["ansatz"].unique())
    variances = sorted(filtered_df["fcc.pulse_params_variance"].unique())

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
                filtered_df["fcc.pulse_params_variance"] == variance
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
            legendgroup=f"circuit_{ansatz}",
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
            legendgroup=f"variance_{variance}",
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
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Coefficient Variance",
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
    filtered_df = df[df["fcc.pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sorted(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes:
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fcc over different seeds for a given distortion
        grouped_df = circuit_df.groupby("fcc.pulse_params_variance").fcc
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
        title="FCC over Pulse Parameter Variances",
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
    filtered_df = df[df["fcc.pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sorted(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes:
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fidelity over different seeds for a given distortion
        grouped_df = circuit_df.groupby("fcc.pulse_params_variance")["fidelity"]
        mean = grouped_df.mean()
        std = grouped_df.std()

        fig.add_scatter(
            x=mean.index,
            y=mean.values,
            error_y=dict(type="data", array=std.values, visible=show_error),
            mode="lines+markers",
            name=f"{circuit_name_to_str(ansatz)}",
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
            ),
            line=dict(color=next(color_it)),
        )

    fig.update_layout(
        title="Trace Distance over Pulse Parameter Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Trace Distance",
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
    filtered_df = df[df["fcc.pulse_params_variance"] <= max_distortion]

    # Get unique circuit types
    ansatzes = sorted(filtered_df["ansatz"].unique())
    color_it = iter(design.prim_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes:
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fidelity over different seeds for a given distortion
        grouped_df = circuit_df.groupby("fcc.pulse_params_variance")["trace-distance"]
        mean = grouped_df.mean()
        std = grouped_df.std()

        fig.add_scatter(
            x=mean.index,
            y=mean.values,
            error_y=dict(type="data", array=std.values, visible=show_error),
            mode="lines+markers",
            name=f"{circuit_name_to_str(ansatz)}",
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
            ),
            line=dict(color=next(color_it)),
        )

    fig.update_layout(
        title="Fidelity over Pulse Parameter Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Trace Distance",
        template=design.template,
        font=dict(size=design.font_size),
    )

    return fig
