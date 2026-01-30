import os
import plotly
import plotly.graph_objects as go
from typing import List
import pandas as pd
from copy import deepcopy

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
    main_colors_lst = plotly.colors.qualitative.Dark2
    sec_colors_lst = plotly.colors.qualitative.Pastel2


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

    for it, fig in enumerate(figures):
        # applying last changes
        fig.update_layout()

        filename = f"{path}{name}-{it}.pdf"
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

    return figures


def coeff_var_over_distortion(df: pd.DataFrame, max_distortion, show_error):
    """
    Given a dataframe with variances for different frequencies
    and different distortions, plot the variances over the frequencies
    and use different traces for the distortions

    Args:
        df (pd.DataFrame): _description_
    """
    fig = go.Figure()

    # Extract frequency indices from column names
    freq_indices = sorted(
        [
            float(col.split("coeff.var.f")[1])
            for col in df.columns
            if col.startswith("coeff.var.f")
        ]
    )

    # Get unique distortion values
    distortions = sorted(df["fcc.pulse_params_variance"].unique())
    color_it = iter(design.main_colors_lst)

    # Create a trace for each distortion level
    for it, distortion in enumerate(distortions):
        distortion = float(distortion)
        if distortion > max_distortion or it > 6:
            break

        # Filter data for this distortion
        distortion_df = df[df["fcc.pulse_params_variance"] == distortion]

        # Calculate mean and std for each frequency
        means = []
        stds = []
        for freq_idx in freq_indices:
            col_name = f"coeff.var.f{freq_idx}"
            means.append(distortion_df[col_name].mean())
            stds.append(distortion_df[col_name].std())

        # Add trace with error bars
        fig.add_scatter(
            x=freq_indices,
            y=means,
            error_y=dict(type="data", array=stds, visible=show_error),
            mode="lines+markers",
            name=f"Variance {distortion}",
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
                # symbol=next(design.symbols_iterator),
            ),
            line=dict(color=next(color_it)),
        )

    fig.update_layout(
        title="Coefficient Variance over Frequency Indices",
        xaxis_title="Frequency Index",
        yaxis_title="Coefficient Variance",
        template=design.template,
        font=dict(size=design.font_size),
    )

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
    color_it = iter(design.main_colors_lst)

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
            name=f"{ansatz}",
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
    color_it = iter(design.main_colors_lst)

    # Create a trace for each circuit type
    for ansatz in ansatzes:
        # Filter data for this circuit type
        circuit_df = filtered_df[filtered_df["ansatz"] == ansatz]

        # average the fidelity over different seeds for a given distortion
        grouped_df = circuit_df.groupby("fcc.pulse_params_variance").fidelity
        mean_fidelity = grouped_df.mean()
        std_fidelity = grouped_df.std()

        fig.add_scatter(
            x=mean_fidelity.index,
            y=mean_fidelity.values,
            error_y=dict(type="data", array=std_fidelity.values, visible=show_error),
            mode="lines+markers",
            name=f"{ansatz}",
            marker=dict(
                size=design.marker_size,
                line=dict(width=design.marker_line_width),
            ),
            line=dict(color=next(color_it)),
        )

    fig.update_layout(
        title="Fidelity over Pulse Parameter Variances",
        xaxis_title="Pulse Parameter Variances",
        yaxis_title="Fidelity",
        template=design.template,
        font=dict(size=design.font_size),
    )

    return fig
