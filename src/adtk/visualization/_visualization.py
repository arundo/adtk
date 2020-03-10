"""
We don't typing the visualization module because there are a lot recursion on
nested tree structure which would be messy if we type rigorously."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.plotting import register_matplotlib_converters

from ..data import to_events, to_labels, validate_events

register_matplotlib_converters()


def plot(
    ts=None,
    anomaly=None,
    curve_group="each",
    ts_linewidth=0.5,
    ts_color=None,
    ts_alpha=1.0,
    ts_marker=".",
    ts_markersize=2,
    match_curve_name=True,
    anomaly_tag="span",
    anomaly_color=None,
    anomaly_alpha=0.3,
    anomaly_marker="o",
    anomaly_markersize=4,
    freq_as_period=True,
    axes=None,
    figsize=None,
    legend=True,
):
    """Plot time series and/or anomalies.

    Parameters
    ----------
    ts: pandas Series or DataFrame, optional
        Time series to plot.

    anomaly: list, pandas Series, DataFrame, or (nested) dict of them, optional
        Anomalies to plot.

        - If list, a list of anomalous events (pandas Timestamp for an
          instantaneous event or 2-tuple of pandas Timestamps for an interval);
        - If pandas Series, a binary series indicating normal/anomalous;
        - If pandas DataFrame, each column is treated independently as a binary
          Series.
        - If (nested) dict, every leaf node (list, Series, or DataFrame) is
          treated independently as above.

    curve_group: str or list, optional
        Groups of curves to be drawn at same plots.

        - If str, 'each' means every dimension is drawn in a separated plot,
          'all' means all dimensions are drawn in the same plot.
        - If list, each element corresponds to a subplot, which is the name of
          time series to plot in this subplot, or a list of names. For example,
          ["A", ("B", "C")] means two subplots, where the first one contain
          series A, while the second one contains series B and C.

        Default: 'each'.

    ts_linewidth: float or dict, optional
        Line width of each time series curve.

        - If float, all curves have the same line width.
        - If dict, the key is series name, the value is line width of that
          series.

        Default: 0.5.

    ts_color: str or dict, optional
        Color of each time series curve.

        - If str, all curves have the same color.
        - If dict, the key is series name, the value is color of that series.
        - If None, color will be assigned automatically.

        Default: None.

    ts_alpha: float or dict, optional
        Opacity of each time series curve.

        - If float, all curves have the same opacity.
        - If dict, the key is series name, the value is opacity of that series.

        Default: 1.0.

    ts_marker: str or dict, optional
        Marker type of each time series curve.

        - If str, all curves have the same marker type.
        - If dict, the key is series name, the value is marker type of that
          series.

        Default: ".".

    ts_markersize: int or dict, optional
        Marker size of each time series curve.

        - If int, all curves have the same marker size.
        - If dict, the key is series name, the value is marker size of that
          series.

        Default: 2.

    match_curve_name: bool, optional
        Whether to plot anomaly with corresponding curve by matching series
        names. If False, plot anomaly with all curves.
        Default: True.

    anomaly_tag: str, or (nested) dict, optional
        Plot anomaly as horizontal spans or markers on curves.

        - If str, either 'span' or 'marker', all anomalies are marked with the
          same type of tag.
        - If (nested) dict, it must have a tree structure identical to or
          smaller than that of (nested) dict argument `anomaly`, which can
          define tags for all leaf nodes in `anomaly`.

        Default: "span".

    anomaly_color: str, or (nested) dict, optional
        Color of each anomaly tag.

        - If str, all anomalies are marked with the same color.
        - If (nested) dict, it must have a tree structure identical to or
          smaller than that of (nested) dict argument `anomaly`, which can
          define colors for all leaf nodes in `anomaly`.
        - If None, color will be assigned automatically.

        Default: None.

    anomaly_alpha: float, or (nested) dict, optional
        Opacity of each anomaly tag. Only used for anomaly drawn as horizontal
        spans.

        - If float, all anomalies are marked with the same opacity.
        - If (nested) dict, it must have a tree structure identical to or
          smaller than that of (nested) dict argument `anomaly`, which can
          define opacity for all leaf nodes in `anomaly`.

        Default: 0.3.

    anomaly_marker: str, or (nested) dict, optional
        Marker type of each anomaly marker. Only used for anomaly drawn as
        markers on curves.

        - If str, all anomalies are marked with the same type of marker.
        - If (nested) dict, it must have a tree structure identical to or
          smaller than that of (nested) dict argument `anomaly`, which can
          define marker types for all leaf nodes in `anomaly`.

        Default: "o".

    anomaly_markersize: int, or (nested) dict, optional
        Marker size of each anomaly marker. Only used for anomaly drawn as
        markers on curves.

        - If int, all anomalies are marked with the same size of marker.
        - If (nested) dict, it must have a tree structure identical to or
          smaller than that of (nested) dict argument `anomaly`, which can
          define marker sizes for all leaf nodes in `anomaly`.

        Default: 4.

    freq_as_period: bool, optional
        Whether to regard time index with regular frequency (i.e. attribute
        `freq` of time index is not None) as time intervals. Only used when
        anomaly is given as binary series.

        For example, DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03',
        '2017-01-04', '2017-01-05'], dtype='datetime64[ns]', freq='D') has
        daily frequency. If freq_as_period=True, each time point in the index
        represents that day (24 hours). Otherwsie, each time point represents
        the instantaneous time instance of 00:00:00 on that day.

        Default: True.

    axes: matplotlib Axes object, or array of Axes objects, optional
        Axes to plot at. The number of Axes objects should be equal to the
        number of plots. If not specified, figure axes will be automatically
        generated. Default: None.

    figsize: tuple, optional
        Size of the figure. If not specified, the size of each subplot is 16 x
        4. Default: None.

    legend: bool, optional
        Whether to show legend in the plot. Default: True.

    Returns
    --------
    matplotlib Axes object or array of Axes objects
        Axes where the plot(s) is drawn.

    """
    # setup style
    plt.style.use("seaborn-whitegrid")

    # initialize color generator
    color_generator = ColorGenerator()

    # plot time series
    if ts is not None:
        # type check for ts
        if isinstance(ts, pd.Series):
            if ts.name is None:
                df = ts.to_frame("Time Series")
            else:
                df = ts.to_frame()
        elif isinstance(ts, pd.DataFrame):
            df = ts.copy()
        else:
            raise TypeError(
                "Argument `ts` must be a pandas Series or DataFrame."
            )

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

        # validate curve groups
        curve2axes = _validate_curve_group(df, curve_group)

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
        for ax in axes:
            ax.xaxis_date()

        # expand ts properties to a dict, if not yet
        ts_color = _assign_properties(ts_color, df)
        ts_linewidth = _assign_properties(ts_linewidth, df, 0.5)
        ts_marker = _assign_properties(ts_marker, df, ".")
        ts_markersize = _assign_properties(ts_markersize, df, 2)
        ts_alpha = _assign_properties(ts_alpha, df, 1.0)

        # plot curves
        _plot_curve(
            df,
            axes,
            curve2axes,
            ts_color,
            ts_linewidth,
            ts_marker,
            ts_markersize,
            ts_alpha,
            color_generator,
        )
    else:  # no time series, just event
        df = pd.DataFrame(dtype=int)
        curve2axes = dict()

        # never try to match curve name, because there is no curve anyway
        match_curve_name = False

        # never try to plot on curve, because there is no curve anyway
        anomaly_tag = "span"

        # setup figure
        if figsize is None:
            figsize = (16, 4)

        # setup axes
        if axes is None:
            _, axes = plt.subplots(figsize=figsize)
        if not isinstance(axes, (list, np.ndarray)):
            axes = [axes]
        for ax in axes:
            ax.xaxis_date()

    # plot anomaly
    if anomaly is not None:
        # validate anomaly
        _validate_anomaly(anomaly)
        # this is for showing a legend even if the series does not have a key
        if isinstance(anomaly, (list, pd.Series)):
            anomaly = {"Anomaly": anomaly}

        # expand tree struct of anomaly properties to match that of `anomaly`
        anomaly_tag = _assign_properties(anomaly_tag, anomaly, "span")
        anomaly_color = _assign_properties(anomaly_color, anomaly)
        anomaly_alpha = _assign_properties(anomaly_alpha, anomaly, 0.3)
        anomaly_marker = _assign_properties(anomaly_marker, anomaly, "o")
        anomaly_markersize = _assign_properties(anomaly_markersize, anomaly, 4)

        # plot anomalies
        _plot_anomaly(
            anomaly,
            axes,
            df,
            curve2axes,
            anomaly_tag,
            anomaly_color,
            anomaly_marker,
            anomaly_markersize,
            anomaly_alpha,
            match_curve_name,
            freq_as_period,
            color_generator,
        )

    # display legend
    if legend and ((ts is not None) or (anomaly is not None)):
        for ax in axes:
            ax.legend()

    return axes


def _validate_curve_group(df, curve_group):
    "Validate curve group, and return inverse map."
    curve2group = {col: set() for col in df.columns}
    for ind, group in enumerate(curve_group):
        if not isinstance(
            group, (list, tuple)
        ):  # this group has a single curve
            if group in set(df.columns):
                curve2group[group].add(ind)
            else:
                raise ValueError(
                    "{} is not a seriers in input `ts`.".format(group)
                )
        else:
            for curve in group:
                if curve in set(df.columns):
                    curve2group[curve].add(ind)
                else:
                    raise ValueError(
                        "{} is not a seriers in input `ts`.".format(curve)
                    )
    return curve2group


def _plot_curve(
    df,
    axes,
    curve2axes,
    ts_color,
    ts_linewidth,
    ts_marker,
    ts_markersize,
    ts_alpha,
    color_generator,
):
    "Plot all curves"
    for col, axes_inds in curve2axes.items():
        color = color_generator.emit(ts_color[col])
        for axes_ind in axes_inds:
            # df[col].plot(
            #     ax=axes[axes_ind],
            #     color=color,
            #     linewidth=ts_linewidth[col],
            #     marker=ts_marker[col],
            #     markersize=ts_markersize[col],
            #     alpha=ts_alpha[col],
            #     label=str(col),
            # )
            axes[axes_ind].plot_date(
                df.index,
                df[col],
                fmt="-",
                color=color,
                linewidth=ts_linewidth[col],
                marker=ts_marker[col],
                markersize=ts_markersize[col],
                alpha=ts_alpha[col],
                label=str(col),
            )


def _plot_anomaly(
    anomaly,
    axes,
    df,
    curve2axes,
    anomaly_tag,
    anomaly_color,
    anomaly_marker,
    anomaly_markersize,
    anomaly_alpha,
    match_curve_name,
    freq_as_period,
    color_generator,
    anomaly_name=None,
    anomaly_label=None,
):
    if isinstance(anomaly, (list, pd.Series)):
        anomaly_color = color_generator.emit(anomaly_color)
        if anomaly_tag == "span":
            # turn anomaly into list, if not yet
            if isinstance(anomaly, pd.Series):
                anomaly = to_events(anomaly, freq_as_period=freq_as_period)
            anomaly = validate_events(anomaly, point_as_interval=True)
            if match_curve_name and (
                anomaly_name in set(df.columns)
            ):  # match found, plot on it
                for axes_ind in curve2axes[anomaly_name]:
                    _add_anomaly_list_to_axes(
                        anomaly,
                        axes[axes_ind],
                        anomaly_color,
                        anomaly_alpha,
                        (
                            anomaly_label
                            if (anomaly_label != anomaly_name)
                            else "Anomaly - {}".format(anomaly_name)
                        ),
                    )
            else:  # not match found or don't match, plot on all
                for ax in axes:
                    _add_anomaly_list_to_axes(
                        anomaly,
                        ax,
                        anomaly_color,
                        anomaly_alpha,
                        (
                            "Anomaly - {}".format(anomaly_name)
                            if (
                                (anomaly_label == anomaly_name)
                                and (anomaly_name in set(df.columns))
                            )
                            else anomaly_label
                        ),
                    )
        elif anomaly_tag == "marker":
            # turn anomaly into binary series, if not yet
            if isinstance(anomaly, list):
                anomaly = to_labels(
                    anomaly, df.index, freq_as_period=freq_as_period
                )
            else:
                try:
                    pd.testing.assert_index_equal(
                        anomaly.index, df.index, check_names=False
                    )
                except AssertionError:
                    raise ValueError(
                        "Series index in argument `anomaly` must be the same "
                        "as the input time series."
                    )
            if match_curve_name and (
                anomaly_name in set(df.columns)
            ):  # match found, plot on it
                for axes_ind in curve2axes[anomaly_name]:
                    _add_anomaly_series_to_curve(
                        anomaly,
                        axes[axes_ind],
                        df[anomaly_name],
                        anomaly_color,
                        anomaly_marker,
                        anomaly_markersize,
                        (
                            anomaly_label
                            if (anomaly_label != anomaly_name)
                            else "Anomaly - {}".format(anomaly_name)
                        ),
                    )
            else:  # not match found or don't match, plot on all
                # hasLegend is an auxilary variable to make sure an anomaly
                # series only appears once in legend in an axes
                hasLegend = [False] * len(axes)
                for curve, axes_inds in curve2axes.items():
                    for axes_ind in axes_inds:
                        _add_anomaly_series_to_curve(
                            anomaly,
                            axes[axes_ind],
                            df[curve],
                            anomaly_color,
                            anomaly_marker,
                            anomaly_markersize,
                            (
                                "Anomaly - {}".format(anomaly_name)
                                if (
                                    (anomaly_label == anomaly_name)
                                    and (anomaly_name in set(df.columns))
                                )
                                else anomaly_label
                            )
                            if not hasLegend[axes_ind]
                            else None,
                        )
                        hasLegend[axes_ind] = True
        else:
            raise ValueError(
                "An anomaly tag must be either 'span' or 'marker'."
            )
    elif isinstance(anomaly, (pd.DataFrame, dict)):
        for key in (
            anomaly.columns
            if isinstance(anomaly, pd.DataFrame)
            else anomaly.keys()
        ):
            _plot_anomaly(
                anomaly[key],
                axes,
                df,
                curve2axes,
                anomaly_tag[key],
                anomaly_color[key],
                anomaly_marker[key],
                anomaly_markersize[key],
                anomaly_alpha[key],
                match_curve_name,
                freq_as_period,
                color_generator,
                anomaly_name=key,
                anomaly_label=(
                    "{} - {}".format(anomaly_name, key)
                    if (anomaly_name is not None)
                    else key
                ),
            )
    else:
        raise TypeError(
            "Argument `anomaly` must be a list, pandas Series, DataFrame, or "
            "a (nested) dict of them."
        )


def _add_anomaly_list_to_axes(
    anomaly, ax, anomaly_color, anomaly_alpha, anomaly_label
):
    "Add a list of anomalous event to an axes as spans"
    for i, event in enumerate(anomaly):
        ax.axvspan(
            xmin=event[0],
            xmax=event[1],
            color=anomaly_color,
            alpha=anomaly_alpha,
            label=(anomaly_label if i == 0 else None),
        )


def _add_anomaly_series_to_curve(
    anomaly,
    ax,
    s,
    anomaly_color,
    anomaly_marker,
    anomaly_markersize,
    anomaly_label,
):
    "Add anomalies represented by a binary series as markers on a curve"
    anomaly_curve = s.loc[anomaly == 1]
    # anomaly_curve.plot(
    #     ax=ax,
    #     linewidth=0,
    #     marker=anomaly_marker,
    #     markersize=anomaly_markersize,
    #     color=anomaly_color,
    #     label=anomaly_label,
    # )
    ax.plot_date(
        anomaly_curve.index,
        anomaly_curve,
        fmt="-",
        linewidth=0,
        marker=anomaly_marker,
        markersize=anomaly_markersize,
        color=anomaly_color,
        label=anomaly_label,
    )


def _validate_anomaly(anomaly):
    "Validate argument `anomaly`."
    if isinstance(anomaly, (list, pd.Series)):
        pass
    elif isinstance(anomaly, pd.DataFrame):
        if anomaly.columns.duplicated().any():
            raise ValueError(
                "DataFrame in argument `anomaly` must have unique column names."
            )
    elif isinstance(anomaly, dict):
        for _, value in anomaly.items():
            _validate_anomaly(value)
    else:
        raise TypeError(
            "Argument `anomaly` must be a list, pandas Series, DataFrame, or "
            "a (nested) dict of them."
        )


def _assign_properties(prop, anomaly, default=None):
    "Expand the tree structure of `prop` to that of `anomaly`"
    if (not isinstance(prop, dict)) and isinstance(
        anomaly, (dict, pd.DataFrame)
    ):
        return {
            key: _assign_properties(prop, anomaly[key])
            for key in (
                anomaly.keys()
                if isinstance(anomaly, dict)
                else anomaly.columns
            )
        }
    elif (not isinstance(prop, dict)) and (
        not isinstance(anomaly, (dict, pd.DataFrame))
    ):
        return prop
    elif isinstance(prop, dict) and (
        not isinstance(anomaly, (dict, pd.DataFrame))
    ):
        raise ValueError("Property dict and anomaly dict are inconsistent.")
    else:  # isinstance(prop, dict) & isinstance(anomaly, (dict, pd.DataFrame))
        if set(prop.keys()) <= set(
            anomaly.keys() if isinstance(anomaly, dict) else anomaly.columns
        ):
            return {
                key: _assign_properties(
                    (prop[key] if (key in prop.keys()) else default),
                    anomaly[key],
                )
                for key in (
                    anomaly.keys()
                    if isinstance(anomaly, dict)
                    else anomaly.columns
                )
            }
        else:
            raise ValueError(
                "Property dict and anomaly dict are inconsistent."
            )


class ColorGenerator:
    """
    Generate color
    """

    def __init__(self):
        self.latest_auto_color = -1

    def emit(self, color=None):
        if color is not None:
            return color
        else:
            self.latest_auto_color += 1
            return "C{}".format(self.latest_auto_color)
