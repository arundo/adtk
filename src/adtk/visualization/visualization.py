import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from adtk.data import to_events, to_labels, validate_events

__all__ = ["plot"]


def plot(
    ts,
    anomaly=None,
    curve_group="each",
    anomaly_tag="span",
    match_curve_name=True,
    anomaly_color=None,
    anomaly_alpha=0.3,
    anomaly_marker="o",
    anomaly_markersize=None,
    freq_as_period=True,
    axes=None,
    figsize=None,
    legend=True,
):
    """Plot time series and anomalies.

    Parameters
    ----------
    ts: pandas Series or DataFrame
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

    anomaly_tag: str, or (nested) dict, optional
        Plot anomaly as horizontal spans or markers on curves.

        - If str, either 'span' or 'marker'
        - If (nested) dict, it must have a tree structure identical to or
          smaller than that of (nested) dict argument `anomaly`, which can
          define tags for all leaf nodes in `anomaly`.

        Default: "span".

        See ??? for examples

    match_curve_name: bool, optional
        Whether to plot anomaly with corresponding curve by matching series
        names. If False, plot anomaly with all curves.
        Default: True.

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

    # TODO: docstring example
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

    # plot curves
    for ind, group in enumerate(curve_group):
        if isinstance(group, str):  # this group has a single curve
            _add_curve_to_axes(df, group, axes[ind])
        else:  # this group has a list of curves
            for curve in group:
                _add_curve_to_axes(df, curve, axes[ind])

    if anomaly is not None:
        # validate anomaly
        _validate_anomaly(anomaly)
        # this is for showing a legend even if the series does not have a key
        if isinstance(anomaly, (list, pd.Series)):
            anomaly = {"Anomaly": anomaly}

        # expand tree structure of input properties to match that of `anomaly`
        anomaly_tag = _assign_properties(anomaly_tag, anomaly)
        anomaly_color = _assign_properties(anomaly_color, anomaly)
        anomaly_alpha = _assign_properties(anomaly_alpha, anomaly)
        anomaly_marker = _assign_properties(anomaly_marker, anomaly)
        anomaly_markersize = _assign_properties(anomaly_markersize, anomaly)

        # plot anomalies
        _add_anomaly_to_plot(
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
        )

    # display legend
    if legend:
        for ax in axes:
            ax.legend()

    return axes


def _validate_curve_group(df, curve_group):
    "Validate curve group, and return inverse map."
    curve2group = {col: set() for col in df.columns}
    for ind, group in enumerate(curve_group):
        if isinstance(group, str):  # this group has a single curve
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


def _add_anomaly_to_plot(
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
    anomaly_name=None,
):
    if isinstance(anomaly, (list, pd.Series)):
        if anomaly_tag == "span":
            # turn anomaly into list, if not yet
            if isinstance(anomaly, pd.Series):
                anomaly = to_events(
                    anomaly,
                    freq_as_period=freq_as_period,
                    merge_consecutive=True,
                )
            else:
                anomaly = validate_events(anomaly, point_as_interval=True)
            if match_curve_name and (
                anomaly_name in set(df.columns)
            ):  # match found, plot on it
                for axes_ind in curve2axes[anomaly_name]:
                    _plot_anomaly_list_to_axes(
                        anomaly,
                        axes[axes_ind],
                        anomaly_color,
                        anomaly_alpha,
                        anomaly_name,
                    )
            else:  # not match found or don't match, plot on all
                for ax in axes:
                    _plot_anomaly_list_to_axes(
                        anomaly, ax, anomaly_color, anomaly_alpha, anomaly_name
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
                    _plot_anomaly_series_to_curve(
                        anomaly,
                        axes[axes_ind],
                        df[anomaly_name],
                        anomaly_color,
                        anomaly_marker,
                        anomaly_markersize,
                        anomaly_name,
                    )
            else:  # not match found or don't match, plot on all
                for curve, axes_inds in curve2axes.items():
                    for axes_ind in axes_inds:
                        _plot_anomaly_series_to_curve(
                            anomaly,
                            axes[axes_ind],
                            df[curve],
                            anomaly_color,
                            anomaly_marker,
                            anomaly_markersize,
                            anomaly_name,
                        )
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
            _add_anomaly_to_plot(
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
                anomaly_name=(
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


def _plot_anomaly_list_to_axes(
    anomaly, ax, anomaly_color, anomaly_alpha, anomaly_name
):
    for i, event in enumerate(anomaly):
        ax.axvspan(
            xmin=event[0],
            xmax=event[1],
            color=anomaly_color,
            alpha=anomaly_alpha,
            label=(anomaly_name if i == 0 else None),
        )


def _plot_anomaly_series_to_curve(
    anomaly,
    ax,
    s,
    anomaly_color,
    anomaly_marker,
    anomaly_markersize,
    anomaly_name,
):
    anomaly_curve = s.loc[anomaly == 1]
    ax.plot(
        anomaly_curve,
        linewidth=0,
        marker=anomaly_marker,
        markersize=anomaly_markersize,
        color=anomaly_color,
        label=anomaly_name,
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


def _assign_properties(prop, anomaly):
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
                    (prop[key] if (key in prop.keys()) else None), anomaly[key]
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
