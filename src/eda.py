''' Utility functions for exploratory data analysis '''
from typing import Union

import pandas
import plotly.express as px
import matplotlib.pyplot as plt
import statsmodels.api as sm

def ts_lineplot_facet(df: pandas.DataFrame,
                      x_var: str,
                      y_var: str,
                      width: int = 800,
                      height: int = 1200,
                      color: str = None,
                      title: str = "",
                      facet_col: str = None,
                      facet_col_wrap: int = 0,
                      facet_col_spacing: float = 0.02):
    '''
    Generate a line plot for a timeseries, if required with color grouping and faceting. Assumes that the df is already
    subset to relevant data points and ordered.
    Args:
        df (pandas.DataFrame): Input data frame, must have the columns specified in x_var, y_var, and color and subset
            to relevant data points and ordered
        x_var (str): Column name to be used as x-axis variable
        y_var (str): Column name to be used as y-axis variable
        width (int, optional): width of plot
        height (int, optional): height of plot
        color (str, optional): Column name to add additional lines in different colors
        title (str, optional): Title to use for the plot
        facet_col (str, optional): Either a name of a column in df. Values from this column are used to assign marks to
            facetted subplots in the horizontal direction.
        facet_col_wrap (int, optional): Maximum number of facet columns. Ignored if 0.
        facet_col_spacing (float, optional):  Spacing between facet cols, in paper units.
    Returns:
        (figure object): The plotly figure
    '''
    # Check if specified values are indeed columns
    if color is not None:
        if facet_col is not None:
            expec_cols = {x_var, y_var, color, facet_col}
        else:
            expec_cols = {x_var, y_var, color}
    else:
        if facet_col is not None:
            expec_cols = {x_var, y_var, facet_col}
        else:
            expec_cols = {x_var, y_var}

    miss_cols = expec_cols.difference({*df.columns})
    if len(miss_cols) != 0:
        raise KeyError(f"Some variables specified do not exist in df. Variables that do not exist: {miss_cols}")

    # Generate plot
    if color is not None:
        if facet_col is not None:
            fig = px.line(df, x=x_var, y=y_var, color=color, title=title, markers=True,
                          facet_col=facet_col, facet_col_wrap=facet_col_wrap,
                          facet_col_spacing=facet_col_spacing, width=width, height=height)
        else:
            fig = px.line(df, x=x_var, y=y_var, color=color, title=title, markers=True)
    else:
        if facet_col is not None:
            fig = px.line(df, x=x_var, y=y_var, color=color, title=title, markers=True,
                          facet_col=facet_col, facet_col_wrap=facet_col_wrap,
                          facet_col_spacing=facet_col_spacing, width=width, height=height)
        else:
            fig = px.line(df, x=x_var, y=y_var, title=title, markers=True, width=width, height=height)
    return fig

def ts_lineplot_facet_row(df: pandas.DataFrame,
                      x_var: str,
                      y_var: str,
                      width: int = 800,
                      height: int = 1200,
                      color: str = None,
                      title: str = "",
                      facet_row: str = None,
                      facet_row_spacing: float = 0.06):
    '''
    Generate a line plot for a timeseries, if required with color grouping and faceting. Assumes that the df is already
    subset to relevant data points and ordered.
    Args:
        df (pandas.DataFrame): Input data frame, must have the columns specified in x_var, y_var, and color and subset
            to relevant data points and ordered
        x_var (str): Column name to be used as x-axis variable
        y_var (str): Column name to be used as y-axis variable
        width (int, optional): width of plot
        height (int, optional): height of plot
        color (str, optional): Column name to add additional lines in different colors
        title (str, optional): Title to use for the plot
        facet_row (str, optional): Either a name of a column in df. Values from this column are used to assign marks to
            facetted subplots in the vertical direction.
        facet_row_spacing (float, optional):  Spacing between facet rows, in paper units.
    Returns:
        (figure object): The plotly figure
    '''
    # Check if specified values are indeed columns
    if color is not None:
        if facet_row is not None:
            expec_cols = {x_var, y_var, color, facet_row}
        else:
            expec_cols = {x_var, y_var, color}
    else:
        if facet_row is not None:
            expec_cols = {x_var, y_var, facet_row}
        else:
            expec_cols = {x_var, y_var}

    miss_cols = expec_cols.difference({*df.columns})
    if len(miss_cols) != 0:
        raise KeyError(f"Some variables specified do not exist in df. Variables that do not exist: {miss_cols}")

    # Generate plot
    if color is not None:
        if facet_row is not None:
            fig = px.line(df, x=x_var, y=y_var, color=color, title=title, markers=True,
                          facet_row=facet_row,
                          facet_row_spacing=facet_row_spacing, width=width, height=height)
        else:
            fig = px.line(df, x=x_var, y=y_var, color=color, title=title, markers=True)
    else:
        if facet_row is not None:
            fig = px.line(df, x=x_var, y=y_var, color=color, title=title, markers=True,
                          facet_row=facet_row,
                          facet_row_spacing=facet_row_spacing, width=width, height=height)
        else:
            fig = px.line(df, x=x_var, y=y_var, title=title, markers=True, width=width, height=height)
    return fig


def ts_analysis_plots(series: pandas.Series,
                      lags: list[int] = None,
                      figsize : tuple[int] = (12, 8),
                      title: str = ""):
    '''
    Generates a simple line plot of the pandas time series (assumes that the index is a time index), an acf, and a pacf
    plot, alongside a Dickey-Fuller test for testing stationarity of the time series.
    Args:
        series (pandas.Series): The time series to plot, must have time index as index
        lags (list[int], optional): The number of lags to use in the acf and pacf plots. If not specified uses
            default from statsmodels.graphics.tsaplots.[..]
        figsize (tuple[int], optional): Figure size as tuple of integeres (w, h)
        title (str, optional): Title of the plot.

    Returns:
        None
    '''

    fig = plt.figure(figsize=figsize)
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))

    series.plot(ax=ts_ax)
    p_value = sm.tsa.stattools.adfuller(series)[1]
    ts_ax.set_title(title + "\n Dickey-Fuller: p={0:.5f}".format(p_value))
    sm.graphics.tsa.plot_acf(series, lags=lags, ax=acf_ax)
    sm.graphics.tsa.plot_pacf(series, lags=lags, ax=pacf_ax)
    plt.tight_layout()


if __name__ == "__main__":
    # dummy data
    data = sm.datasets.sunspots.load_pandas()
    ts_sun = data.data.set_index('YEAR').SUNACTIVITY

    ts_analysis_plots(ts_sun)