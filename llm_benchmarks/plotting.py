import glob
from typing import List

import pandas as pd
import plotly.express as px


def plot_model_inference_speed(
    model_name: str,
    filters: dict,
    grouping_columns: List,
    colors: dict,
    title: str,
    save_path: str,
    results_dir: str = "./results",
    width: int = 1000,
    height: int = 600,
    scale: int = 5,
) -> None:
    # Concatenate CSV files
    try:
        df = pd.concat(
            [pd.read_csv(f) for f in glob.glob(f"{results_dir}/*{model_name}*.csv")],
            ignore_index=True,
        )
    except ValueError:
        print(f"No CSV files found in {results_dir} for model {model_name}.")
        return

    # Apply filters based on provided dictionary
    for column, bounds in filters.items():
        lower, upper = bounds
        df = df[(df[column] > lower) & (df[column] < upper)]

    # Make interaction feature
    if len(grouping_columns) == 1:
        df["grouping"] = df[grouping_columns[0]]
    elif len(grouping_columns) == 2:
        df["grouping"] = df[grouping_columns[0]] + "_" + df[grouping_columns[1]]
    else:
        raise ValueError("grouping_columns must be a list of length 1 or 2")

    # Plot
    fig = px.scatter(
        df,
        x="output_tokens",
        y="tokens_per_second",
        color="grouping",
        color_discrete_map=colors,
        trendline="ols",
        trendline_options=dict(log_x=True),
    )
    fig.update_traces(marker=dict(size=12))
    fig.update_layout(xaxis_title="output_tokens", yaxis_title="tokens_per_second", title="Scatter Plot")
    fig.update_layout(title=title)
    fig.show()

    # Save
    fig.write_image(save_path, width=width, height=height, scale=scale)
