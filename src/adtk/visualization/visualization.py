import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

__all__ = ["plot"]

"""
Scenarios:

Series -> detector1d(return_list=True) --> list of events
Series -> detector1d(return_list=False) --> binary Series
Series -> pipe(return_list=True, return_intermediate=True) --> dict of lists of events
Series -> pipe(return_list=False, return_intermediate=True) --> dict of binary Series
DF -> detectorhd(return_list=True) --> list of events
DF -> detectorhd(return_list=False) --> binary Series
DF -> detector1d(return_list=True) --> dict of lists (keys == columns)
DF -> detector1d(return_list=False) --> DF (columns == columns)
DF -> pipe(return_list=True, return_intermediate=True) --> dict of (lists of events or dicts of lists of events)
DF -> pipe(return_list=False, return_intermediate=True) --> dict of (Series or DataFrame)

"""


def plot(
    ts,
    anomaly_true=None,
    anomaly_pred=None,
    title=None,
    axes=None,
    figsize=None,
    ts_linewidth=0.5,
    ts_alpha=0.5,
    ts_color=None,
    ts_marker=".",
    ts_markersize=1,
    at_alpha=0.5,
    at_color="red",
    at_marker_on_curve=False,
    at_marker="v",
    at_markersize=3,
    ap_alpha=0.5,
    ap_color="green",
    ap_marker_on_curve=False,
    ap_marker="o",
    ap_markersize=3,
    freq_as_period=True,
    curve_group="each",
    legend=True,
):
    """Plot time series and anomalies.

    Parameters
    ----------
    ts: pandas Series or DataFrame
        Time series to plot.

    anomaly_true: list, pandas Series, dict, or pandas DataFrame, optional
        True anomalies.

        - If list, a list of known anomalous event time (pandas Timestamp or
          2-tuple of pandas Timestamps);
        - If pandas Series, a binary series indicating normal/anomalous;
        - If dict, a dict of lists or Series where each key-value pair is
          regarded as a type of anomaly.
        - If pandas DataFrame, each column is regarded as a type of anomaly.

    anomaly_pred: list, pandas Series, dict, or pandas DataFrame, optional
        Predicted anomalies.

        - If list, a list of known anomalous event time (pandas Timestamp or
          2-tuple of pandas Timestamps);
        - If pandas Series, a binary series indicating normal/anomalous;
        - If dict, a dict of lists or Series where each key-value pair is
          regarded as a type of anomaly.
        - If pandas DataFrame, each column is regarded as a type of anomaly.

    curve_group: str or list, optional
        Groups of curves to be drawn at same plots.

        - If str, 'each' means every dimension is drawn in a separated plot,
          'all' means all dimensions are drawn in the same plot.
        - If list, each element corresponds to a subplot, which is the name of
          time series to plot in this subplot, or a list of names. For example,
          ["A", ("B", "C")] means two subplots, where the first one contain
          series A, while the second one contains series B and C.

        Default: 'each'.

    anomaly_group_method

    axes: matplotlib Axes object, or list of Axes objects, optional
        Axes to plot at. The number of Axes objects should be equal to the
        number of plots. Default: None.

    figsize: tuple, optional
        Size of the figure. Default: None.

    legend: bool, optional
        Whether to show legend in the plot. Default: True.

    Returns
    --------
    matplotlib Axes object or list
        Axes where the plot(s) is drawn.

    """
    plt.style.use("seaborn-whitegrid")

    # TODO: ts as None

    # type check for ts
    if isinstance(ts, pd.Series):
        if ts.name is None:
            df = ts.to_frame("Time Series")
        else:
            df = ts.to_frame()
    elif isinstance(ts, pd.DataFrame):
        df = ts.copy()
    else:
        raise TypeError("Argument `ts` must be a pandas Series or DataFrame.")

    # check series index
    if not isinstance(df.index, pd.DatetimeIndex):
        raise TypeError(
            "Index of the input time series must be a pandas "
            "DatetimeIndex object."
        )

    # check duplicated column names
    if df.columns.duplicated().any():
        raise ValueError("Input DataFrame must have unique column names.")

    # set up curve groups
    if curve_group == "each":
        curve_group = list(df.columns)
    elif curve_group == "all":
        curve_group = [tuple(df.columns)]

    # set up default figure size
    if figsize is None:
        figsize = (16, 4 * len(curve_group))

    # setup axes
    if axes is None:
        _, axes = plt.subplots(
            nrows=len(curve_group), figsize=figsize, sharex=True
        )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    for ind, group in enumerate(curve_group):
        if isinstance(group, str):
            _add_curve_to_axes(df, group, axes[ind])
        else:
            for curve in group:
                _add_curve_to_axes(df, curve, axes[ind])

    # display legend
    if legend:
        for ax in axes:
            ax.legend()

    return axes


def _add_curve_to_axes(df, curve_name, axes):
    """
    Add a curve to an axes
    """
    axes.plot(
        df.index,
        df[curve_name],
        color="C{}".format(list(df.columns).index(curve_name)),
        label=curve_name,
    )

