"""Module for visualization."""

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from ..data import to_events, to_labels

from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()

__all__ = ["plot"]


def _plot_anomaly_as_span(
    ax,
    anomaly,
    min_datetime,
    max_datetime,
    anomaly_prefix,
    alpha,
    color,
    freq_as_period,
):
    if isinstance(anomaly, list):
        hasLegend = False
        for a in anomaly:
            if isinstance(a, tuple):
                if ((a[0] <= max_datetime) & (a[0] >= min_datetime)) or (
                    (a[1] <= max_datetime) & (a[1] >= min_datetime)
                ):
                    if a[0] <= a[1]:
                        if not hasLegend:
                            ax.axvspan(
                                a[0],
                                a[1],
                                color=color,
                                alpha=alpha,
                                label="{}anomalies".format(anomaly_prefix),
                            )
                            hasLegend = True
                        else:
                            ax.axvspan(a[0], a[1], color=color, alpha=alpha)
            else:
                if (a <= max_datetime) & (a >= min_datetime):
                    if not hasLegend:
                        ax.axvspan(
                            a,
                            a,
                            color=color,
                            alpha=alpha,
                            label="{}anomalies".format(anomaly_prefix),
                        )
                        hasLegend = True
                    else:
                        ax.axvspan(a, a, color=color, alpha=alpha)
    elif isinstance(anomaly, pd.Series):
        _plot_anomaly_as_span(
            ax,
            to_events(anomaly, freq_as_period),
            min_datetime,
            max_datetime,
            anomaly_prefix,
            alpha,
            color,
            freq_as_period,
        )
    elif isinstance(anomaly, dict):
        if isinstance(next(iter(anomaly.values())), pd.Series):
            _plot_anomaly_as_span(
                ax,
                {
                    key: to_events(value, freq_as_period)
                    for key, value in anomaly.items()
                },
                min_datetime,
                max_datetime,
                anomaly_prefix,
                alpha,
                color,
                freq_as_period,
            )
        else:
            for key, ano in anomaly.items():
                for counter, a in enumerate(ano):
                    if isinstance(a, tuple):
                        if (
                            (a[0] <= max_datetime) & (a[0] >= min_datetime)
                        ) | ((a[1] <= max_datetime) & (a[1] >= min_datetime)):
                            if counter == 0:
                                ax.axvspan(
                                    a[0],
                                    a[1],
                                    color=color
                                    if not isinstance(color, dict)
                                    else color[key],
                                    alpha=alpha,
                                    label="{}{}".format(anomaly_prefix, key),
                                )
                            else:
                                ax.axvspan(
                                    a[0],
                                    a[1],
                                    color=color
                                    if not isinstance(color, dict)
                                    else color[key],
                                    alpha=alpha,
                                )
                    else:
                        if (a <= max_datetime) & (a >= min_datetime):
                            if counter == 0:
                                ax.axvspan(
                                    a,
                                    a,
                                    color=color
                                    if not isinstance(color, dict)
                                    else color[key],
                                    alpha=alpha,
                                    label="{}{}".format(anomaly_prefix, key),
                                )
                            else:
                                ax.axvspan(
                                    a,
                                    a,
                                    color=color
                                    if not isinstance(color, dict)
                                    else color[key],
                                    alpha=alpha,
                                )
    else:
        raise TypeError(
            "Anomaly must be a list, a pandas Series, or a dict of lists or Series."
        )


def _plot_anomaly_on_curve(
    ax,
    curve,
    anomaly,
    min_datetime,
    max_datetime,
    anomaly_prefix,
    alpha,
    color,
    marker,
    markersize,
    freq_as_period,
):
    if isinstance(anomaly, list) or (
        isinstance(anomaly, dict)
        and isinstance(next(iter(anomaly.values())), list)
    ):
        labels = to_labels(
            anomaly,
            curve.index[
                (curve.index >= min_datetime) & (curve.index <= max_datetime)
            ],
            freq_as_period,
        )
    else:
        labels = anomaly

    if isinstance(labels, pd.Series):
        curve = curve.copy()
        ax.plot(
            curve[labels == 1].index,
            curve[labels == 1],
            linewidth=0,
            alpha=alpha,
            marker=marker,
            color=color,
            markersize=markersize,
            label="{}anomalies".format(anomaly_prefix),
        )
    else:
        for col, s in labels.items():
            curve = curve.copy()
            ax.plot(
                curve[s == 1].index,
                curve[s == 1],
                linewidth=0,
                alpha=alpha,
                marker=(
                    marker if not isinstance(marker, dict) else marker[col]
                ),
                color=(color if not isinstance(color, dict) else color[col]),
                markersize=(
                    markersize
                    if not isinstance(markersize, dict)
                    else markersize[col]
                ),
                label="{}{}".format(anomaly_prefix, str(col)),
            )


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
):
    """Plot time series and anomalies.

    Parameters
    ----------
    ts: pandas Series or DataFrame
        Time series to plot.

    anomaly_true: list, pandas Series, or dict, optional
        Known anomalies.
        If list, a list of known anomalous event time (pandas Timestamp or
        2-tuple of pandas Timestamps);
        If pandas Series, a binary series indicating normal/anomalous;
        If dict, a dict of lists or Series where each key-value pair is
        regarded as a type of anomaly.

    anomaly_pred: list, pandas Series, or dict, optional
        Predicted anomalies.
        If list, a list of known anomalous event time (pandas Timestamp or
        2-tuple of pandas Timestamps);
        If pandas Series, a binary series indicating normal/anomalous;
        If dict, a dict of lists or Series where each key-value pair is
        regarded as a type of anomaly.

    title: str, optional
        Title of the plot.

    axes: matplotlib axes object, or list of axes objects, optional
        Axes to plot at. The number of axes objects should be equal to the
        number of plots.

    figsize: tuple, optional
        Size of the figure.

    ts_linewidth: float, optional
        Line width of time series curves. Default: 0.5.

    ts_alpha: float, optional
        Transparency of time series curves. Default: 0.5.

    ts_color: str or dict, optional
        Color of time series curves. If a dict, keys are names of series.
        If not given, the program will assign colors automatically. Default:
        None.

    ts_marker: str or dict, optional
        Marker of time series curves. If a dict, keys are names of series.
        Default: ".".

    ts_markersize: float or dict, optional
        Marker size of time series curves. If a dict, keys are names of series.
        Default: 1.

    at_alpha: float, optional
        Transparency of known anomaly markers. Default: 0.5.

    at_color: str or dict, optional
        Color of known anomaly markers. If a dict, keys are names of anomaly
        types. Default: "red".

    at_marker_one_curve: bool, optional
        Whether to plot known anomaly on the curve. Note that it may miss
        anomalies that does not happen at exact time points along the time
        series. Default: False.

    at_marker: str or dict, optional
        Marker of known anomalies if plot on curve. If a dict, keys are names
        of anomaly types. Default: "v".

    at_markersize: float or dict, optional
        Marker size of known anomalies if plot on curve. If a dict, keys are
        names of anomaly types. Default: 3.

    ap_alpha: float, optional
        Transparency of detected anomaly markers. Default: 0.5.

    ap_color: str or dict, optional
        Color of detected anomaly markers. If a dict, keys are names of anomaly
        types. Default: "green".

    ap_marker_one_curve, bool, optional
        Whether to plot detected anomaly on the curve. Note that it may miss
        anomalies that does not happen at exact time points along the time
        series. Default: False.

    ap_marker: str or dict, optional
        Marker of detected anomalies if plot on curve. If a dict, keys are
        names of anomaly types. Default: "o".

    ap_markersize: float or dict, optional
        Marker size of detected anomalies if plot on curve. If a dict, keys are
        names of anomaly types. Default: 3.

    freq_as_period: bool, optional
        Whether to regard time stamps following regular frequency as time
        spans. E.g. time index [2010-01-01, 2010-02-01, 2010-03-01, 2010-04-01,
        2010-05-01] follows monthly frequency, and each time stamp represents
        that month if freq_as_period is True. Otherwsie, each time stamp
        represents the time point 00:00:00 on the first day of that month. This
        is only used to determine anomaly markers when marker_on_curve is on.
        Default: True.

    curve_group: str or nested list of int, optional
        Groups of curves to be drawn at same plots.

        If nested list, for exmaple, [[0], [1,2,5], [4,6]] means dimension #0
        of time series is drawn separated, while dimensions #1, #2, #5 are
        drawn together in the second plot, and dimensions #4 and #6 are drawn
        togehter in the third plot.

        If str, 'each' means every dimension is drawn in a separated plot,
        'all' means all dimensions are drawn in the same plot.

        Default: 'each'.

    Returns
    --------
        matplotlib axes object or list
            Axes where the plot(s) is drawn.

    """

    df = ts
    if isinstance(df, pd.Series):
        df = df.to_frame()
    num_col = len(df.columns)
    if curve_group == "each":
        curve_group = [[i] for i in range(num_col)]
    elif curve_group == "all":
        curve_group = [list(range(num_col))]

    sns.set_style("whitegrid")

    if figsize is None:
        figsize = (16, 4 * len(curve_group))

    if axes is None:
        _, axes = plt.subplots(
            nrows=len(curve_group), figsize=figsize, sharex=True
        )
    if not isinstance(axes, (list, np.ndarray)):
        axes = [axes]

    if any([len(cp) > 1 for cp in curve_group]) & (
        at_marker_on_curve | ap_marker_on_curve
    ):
        at_marker_on_curve = False
        ap_marker_on_curve = False
        warnings.warn(
            "marker_on_curve is automatically switched off as at least one "
            "subplot contains multiple curves.",
            UserWarning,
        )

    for counter, ax in enumerate(axes):
        for cc in df.columns[curve_group[counter]]:
            ax.plot(
                df.index,
                df[cc],
                color=(
                    ts_color
                    if not isinstance(ts_color, dict)
                    else ts_color[cc]
                ),
                alpha=ts_alpha,
                linewidth=ts_linewidth,
                marker=(
                    ts_marker
                    if not isinstance(ts_marker, dict)
                    else ts_marker[cc]
                ),
                markersize=(
                    ts_markersize
                    if not isinstance(ts_markersize, dict)
                    else ts_markersize[cc]
                ),
                label=cc,
            )
        min_datetime = (
            df[df.columns[curve_group[counter]]].dropna(how="all").index.min()
        )
        max_datetime = (
            df[df.columns[curve_group[counter]]].dropna(how="all").index.max()
        )

        if anomaly_true is not None:
            if not at_marker_on_curve:
                _plot_anomaly_as_span(
                    ax,
                    anomaly=anomaly_true,
                    min_datetime=min_datetime,
                    max_datetime=max_datetime,
                    anomaly_prefix="Known ",
                    alpha=at_alpha,
                    color=at_color,
                    freq_as_period=freq_as_period,
                )
            else:
                _plot_anomaly_on_curve(
                    ax,
                    curve=df[df.columns[curve_group[counter][0]]],
                    anomaly=anomaly_true,
                    min_datetime=min_datetime,
                    max_datetime=max_datetime,
                    anomaly_prefix="Known ",
                    alpha=at_alpha,
                    color=at_color,
                    marker=at_marker,
                    markersize=at_markersize,
                    freq_as_period=freq_as_period,
                )

        if anomaly_pred is not None:
            if not ap_marker_on_curve:
                _plot_anomaly_as_span(
                    ax,
                    anomaly=anomaly_pred,
                    min_datetime=min_datetime,
                    max_datetime=max_datetime,
                    anomaly_prefix="Detected ",
                    alpha=ap_alpha,
                    color=ap_color,
                    freq_as_period=freq_as_period,
                )
            else:
                _plot_anomaly_on_curve(
                    ax,
                    curve=df[df.columns[curve_group[counter][0]]],
                    anomaly=anomaly_pred,
                    min_datetime=min_datetime,
                    max_datetime=max_datetime,
                    anomaly_prefix="Detected ",
                    alpha=ap_alpha,
                    color=ap_color,
                    marker=ap_marker,
                    markersize=ap_markersize,
                    freq_as_period=freq_as_period,
                )

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel("")
        ax.legend()

    return axes if len(axes) > 1 else axes[0]
