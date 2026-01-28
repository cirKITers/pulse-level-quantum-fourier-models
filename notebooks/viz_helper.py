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
    marker_size = 18
    marker_line_width = 1
    marker_a_opacity = 1.0
    marker_b_opacity = 1.0
    marker_a_style = "x"
    marker_a_color = "#009682"
    marker_b_style = "x"
    marker_b_color = "#DF9B1B"
    legend_color = "#002D4C"
    colorscale = "Sunset"
    symbols_iterator = iter(
        ["circle", "square", "diamond", "cross", "x", "triangle-up", "hexagon", "star"]
    )
    main_colors_it = iter(plotly.colors.qualitative.Dark2)
    sec_colors_it = iter(plotly.colors.qualitative.Pastel2)


def save_figures(
    figures: List[go.Figure],
    name: str,
    experiment_id: str,
    hash: str,
    showlegend: bool = True,
    scale: float = 1,
):
    # use the same hashing strategy as in data_helper
    path = f"results/{experiment_id}/{hash}/"
    os.makedirs(path, exist_ok=True)

    for it, fig in enumerate(figures):
        # applying last changes
        fig.update_layout(showlegend=showlegend)

        filename = f"{path}{name}-{it}.pdf"
        print(f"Saving figure to {filename}")
        fig.write_image(filename, scale=scale)


def coeff_var_over_distortion(df: pd.DataFrame):
    """
    Given a dataframe with variances for different frequencies
    and different distortions, plot the variances over the frequencies
    and use different traces for the distortions

    Args:
        df (pd.DataFrame): _description_
    """
    pass


def fcc_over_distortion(df: pd.DataFrame):
    """
    Given a dataframe with fccs for different distortions,
    plot the fcc over the distortions

    Args:
        df (pd.DataFrame): _description_
    """

    fig = go.Figure()

    # average the fcc over different seeds for a given distortion
    grouped_df = df.groupby("fcc.pulse_params_variance").fcc
    mean_fcc = grouped_df.mean()
    std_fcc = grouped_df.std()

    fig.add_scatter(
        x=mean_fcc.index,
        y=mean_fcc.values,
        error_y=dict(type="data", array=std_fcc.values, visible=True),
        mode="lines+markers",
        marker=dict(
            size=design.marker_size,
            line=dict(width=design.marker_line_width),
            color=design.marker_a_color,
            opacity=design.marker_a_opacity,
        ),
    )

    fig.update_layout(
        title="FCC over Pulse Parameter Variances",
        xaxis_title="Variance",
        yaxis_title="FCC",
        template=design.template,
        font=dict(size=design.font_size),
    )

    return fig
