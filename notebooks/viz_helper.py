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
        print(f"Saving figure to {filename}.pdf")
        fig.write_image(filename, scale=scale)
