import sys
import pandas as pd
import plotly.graph_objects as go
import ml_tools as tools
import argparse
from plotly.subplots import make_subplots


HOUSE = "Hogwarts House"


def plot_grouped_histogram(df):
    num_cols = len(df.select_dtypes(include=[float, int]).columns)
    fig = make_subplots(
        rows=num_cols + 1, cols=1, row_heights=[0.1] + [0.9 / num_cols] * num_cols
    )

    numeric_cols = df.select_dtypes(include=[float, int]).columns
    subplot_titles = [col for col in numeric_cols if col != HOUSE]
    for idx, title in enumerate(subplot_titles, start=1):
        fig.add_trace(
            go.Scatter(
                x=[0],
                y=[0],
                mode="markers+text",
                marker=dict(size=0),
                text=title,
                showlegend=False,
            ),
            row=idx,
            col=1,
        )

    for idx, column in enumerate(numeric_cols):
        if column != HOUSE:
            cleaned = tools.remove_empty_fields(df[column])
            if len(cleaned) == 0:
                continue
            hist_slytherin = go.Histogram(
                x=df[df[HOUSE] == "Slytherin"][column],
                name="Slytherin",
                marker=dict(color="green"),
                opacity=0.4,
            )
            hist_gryffindor = go.Histogram(
                x=df[df[HOUSE] == "Gryffindor"][column],
                name="Gryffindor",
                marker=dict(color="red"),
                opacity=0.4,
            )
            hist_ravenclaw = go.Histogram(
                x=df[df[HOUSE] == "Ravenclaw"][column],
                name="Ravenclaw",
                marker=dict(color="cyan"),
                opacity=0.4,
            )
            hist_hufflepuff = go.Histogram(
                x=df[df[HOUSE] == "Hufflepuff"][column],
                name="Hufflepuff",
                marker=dict(color="gold"),
                opacity=0.4,
            )

            fig.add_trace(hist_slytherin, row=idx + 1, col=1)
            fig.add_trace(hist_gryffindor, row=idx + 1, col=1)
            fig.add_trace(hist_ravenclaw, row=idx + 1, col=1)
            fig.add_trace(hist_hufflepuff, row=idx + 1, col=1)

    fig.update_layout(
        title_text="Histograms of Hogwarts Houses",
        height=600 * (num_cols + 1),
        showlegend=False,
        barmode="overlay",
    )

    fig.show()


def plot_individual_histogram(df):
    numeric_cols = df.select_dtypes(include=[float, int]).columns
    for column in numeric_cols:
        if column != HOUSE:
            fig = go.Figure()

            cleaned = tools.remove_empty_fields(df[column])
            if len(cleaned) == 0:
                continue
            hist_slytherin = go.Histogram(
                x=df[df[HOUSE] == "Slytherin"][column],
                name="Slytherin",
                marker=dict(color="green"),
                opacity=0.4,
            )
            hist_gryffindor = go.Histogram(
                x=df[df[HOUSE] == "Gryffindor"][column],
                name="Gryffindor",
                marker=dict(color="red"),
                opacity=0.4,
            )
            hist_ravenclaw = go.Histogram(
                x=df[df[HOUSE] == "Ravenclaw"][column],
                name="Ravenclaw",
                marker=dict(color="cyan"),
                opacity=0.4,
            )
            hist_hufflepuff = go.Histogram(
                x=df[df[HOUSE] == "Hufflepuff"][column],
                name="Hufflepuff",
                marker=dict(color="gold"),
                opacity=0.4,
            )

            fig.add_trace(hist_slytherin)
            fig.add_trace(hist_gryffindor)
            fig.add_trace(hist_ravenclaw)
            fig.add_trace(hist_hufflepuff)

            fig.update_layout(
                title_text=f"Histogram of {column} for Hogwarts Houses",
                showlegend=True,
                barmode="overlay",
            )
            fig.show()


if __name__ == "__main__":
    try:

        parser = argparse.ArgumentParser(description="Plot all histograms of a dataset")
        parser.add_argument("csv_file", type=str, help="csv file to plot")
        parser.add_argument(
            "-g", "--group", action="store_true", help="group histograms"
        )
        parser.add_argument(
            "-n", "--normalize", action="store_true", help="normalize column"
        )
        args = parser.parse_args()
        try:
            tools.is_valid_path(args.csv_file)
        except Exception as e:
            sys.exit(e)
        df = pd.read_csv(args.csv_file).drop(columns=["Index"])
        if args.normalize:
            df = tools.normalize_df(df)
        if args.group:
            plot_grouped_histogram(df)
        else:
            plot_individual_histogram(df)
    except Exception as e:
        print(e)
        exit(1)
