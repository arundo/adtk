"""Module is for data (time series and anomaly list) processing.
"""

from functools import reduce
from math import gcd

import numpy as np
import pandas as pd

__all__ = [
    "validate_series",
    "to_events",
    "to_labels",
    "expand_events",
    "validate_events",
    "resample",
    "split_train_test",
]


def validate_series(ts, check_freq=True, check_categorical=False):
    """Validate time series.

    This process will check some common critical issues that may cause problems
    if anomaly detection is performed to time series without fixing them. The
    function will automatically fix some of them, while it will raise errors
    when detect others.

    Issues will be checked and automatically fixed include:

    - Time index is not monotonically increasing;
    - Time index contains duplicated time stamps (fix by keeping first values);
    - (optional) Time index attribute `freq` is missed;
    - (optional) Time series include categorical (non-binary) label columns
      (to fix by converting categorical labels into binary indicators).

    Issues will be checked and raise error include:

    - Wrong type of time series object (must be pandas Series or DataFrame);
    - Wrong type of time index object (must be pandas DatetimeIndex).

    Parameters
    ----------
    ts: pandas Series or DataFrame
        Time series to be validated.

    check_freq: bool, optional
        Whether to check time index attribute `freq` is missed. Default: True.

    check_categorical: bool, optional
        Whether to check time series include categorical (non-binary) label
        columns. Default: False.

    Returns
    -------
    pandas Series or DataFrame
        Validated time series.

    """

    ts = ts.copy()

    # check input type
    if not isinstance(ts, (pd.Series, pd.DataFrame)):
        raise TypeError("Input is not a pandas Series or DataFrame object")

    # check index type
    if not isinstance(ts.index, pd.DatetimeIndex):
        raise TypeError(
            "Index of time series must be a pandas DatetimeIndex object."
        )

    # check duplicated
    if any(ts.index.duplicated(keep="first")):
        ts = ts[ts.index.duplicated(keep="first") == False]

    # check sorted
    if not ts.index.is_monotonic_increasing:
        ts.sort_index(inplace=True)

    # check time step frequency
    if check_freq:
        if ts.index.freq is None:
            if ts.index.inferred_freq is not None:
                ts = ts.asfreq(ts.index.inferred_freq)
            elif len(np.unique(np.diff(ts.index))) == 1:
                ts = ts.asfreq(pd.Timedelta(ts.index[1] - ts.index[0]))

    # convert categorical labels into binary indicators
    if check_categorical:
        if isinstance(ts, pd.DataFrame):
            ts = pd.get_dummies(ts)
        if isinstance(ts, pd.Series):
            seriesName = ts.name
            ts = pd.get_dummies(
                ts.to_frame(),
                prefix="" if seriesName is None else seriesName,
                prefix_sep="" if seriesName is None else "_",
            )
            if len(ts.columns) == 1:
                ts = ts[ts.columns[0]]
                ts.name = seriesName

    return ts


def validate_events(event_list, point_as_interval=False):
    """Validate event list.

    This process will check some common issues in an event list (a list of time
    windows), including invalid time window, overlapped or consecutive time
    windows, unsorted events, etc.

    Parameters
    ----------
    event_list: list of pandas Timestamp 2-tuples
        Start and end of original (unmerged) time windows. Every window is
        regarded as a closed interval.

    point_as_interval: bool, optional
        Whether to return a singular time point as a close interval. Default:
        False.

    Returns
    -------
    list of pandas Timestamp 2-tuples:
        Start and end of merged events.

    """
    if not isinstance(event_list, list):
        raise TypeError("Argument `event_list` must be a list.")
    for event in event_list:
        if not (
            isinstance(event, pd.Timestamp)
            or (
                isinstance(event, tuple)
                and (len(event) == 2)
                and all([isinstance(event[i], pd.Timestamp) for i in [0, 1]])
            )
        ):
            raise TypeError(
                "Every event in the list must be a pandas Timestamp, "
                "or a 2-tuple of Timestamps."
            )

    time_window_ends = []
    time_window_type = []
    for time_window in event_list:
        if isinstance(time_window, tuple):
            if time_window[0] <= time_window[1]:
                time_window_ends.append(time_window[0])
                time_window_type.append(+1)
                time_window_ends.append(time_window[1])
                time_window_type.append(-1)
        else:
            time_window_ends.append(time_window)
            time_window_type.append(+1)
            time_window_ends.append(time_window)
            time_window_type.append(-1)
    time_window_end_series = pd.Series(
        time_window_type, index=pd.DatetimeIndex(time_window_ends)
    )
    time_window_end_series.sort_index(kind="mergesort", inplace=True)
    time_window_end_series = time_window_end_series.cumsum()
    status = 0
    merged_event_list = []
    for t, v in time_window_end_series.iteritems():
        if (status == 0) and (v > 0):
            start = t
            status = 1
        if (status == 1) and (v <= 0):
            end = t
            merged_event_list.append([start, end])
            status = 0
    for i in range(1, len(merged_event_list)):
        this_start = merged_event_list[i][0]
        this_end = merged_event_list[i][1]
        last_start = merged_event_list[i - 1][0]
        last_end = merged_event_list[i - 1][1]
        if last_end + pd.Timedelta("1ns") >= this_start:
            merged_event_list[i] = [last_start, this_end]
            merged_event_list[i - 1] = None
    merged_event_list = [
        w[0] if (w[0] == w[1] and not point_as_interval) else tuple(w)
        for w in merged_event_list
        if w is not None
    ]
    return merged_event_list


def to_events(labels, freq_as_period=True, merge_consecutive=None):
    """Convert binary label series to event list(s).

    Parameters
    ----------
    labels: pandas Series or DataFrame
        Binary series of anomaly labels. If a DataFrame, each column is
        regarded as an independent type of anomaly.

    freq_as_period: bool, optional
        Whether to regard time index with regular frequency (i.e. attribute
        `freq` of time index is not None) as time spans.
        E.g. DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03',
        '2017-01-04', '1970-01-05'], dtype='datetime64[ns]', freq='D') has
        daily frequency. If freq_as_period=True, each element represents that
        day. Otherwsie, each time element represents the instantaneous time
        stamp 00:00:00 on that day.
        Default: True.

    merge_consecutive: bool, optional
        Whether to merge consecutive events into a time window. When the option
        is on, if input time index has regular frequency (i.e. attribute `freq`
        of time index is not None) and freq_as_period=True, a merged event ends
        at the end of last period; otherwise, it ends at the last instantaneous
        time point. If the option is not specified, it is on automatically if
        the input time index has regular frequency and freq_as_period=True, and
        is off otherwise. Default: None.

    Returns
    -------
    list or dict
        - If input is a Series, output is a list of time instants or periods.
        - If input is a DataFrame, every column is treated as an independent
          binary series, and output is a dict where keys are column names and
          values are corresponding event lists.

        A time instant is a pandas Timestamp object, while a time period is a
        2-tuple of Timestamp objects that is regarded as a closed interval.

    """

    if isinstance(labels, pd.Series):
        labels = validate_series(
            labels, check_freq=False, check_categorical=False
        )
        labels = labels == 1

        if merge_consecutive is None:
            if freq_as_period and (labels.index.freq is not None):
                merge_consecutive = True
            else:
                merge_consecutive = False

        if not merge_consecutive:
            if freq_as_period and (labels.index.freq is not None):
                period_end = pd.date_range(
                    start=labels.index[1],
                    periods=len(labels.index),
                    freq=labels.index.freq,
                ) - pd.Timedelta("1ns")
                return list(
                    zip(list(labels.index[labels]), list(period_end[labels]))
                )
            else:
                return list(labels.index[labels])
        else:
            labels_values = labels.values.astype(int).reshape(-1, 1)
            mydiff = np.vstack(
                [
                    labels_values[0, :] - 0,
                    np.diff(labels_values, axis=0),
                    0 - labels_values[-1, :],
                ]
            )
            starts = np.argwhere(mydiff == 1)
            ends = np.argwhere(mydiff == -1)
            if freq_as_period and (labels.index.freq is not None):
                period_end = pd.date_range(
                    start=labels.index[1],
                    periods=len(labels.index),
                    freq=labels.index.freq,
                ) - pd.Timedelta("1ns")
                return [
                    (
                        pd.Timestamp(labels.index[start]),
                        pd.Timestamp(period_end[end - 1]),
                    )
                    for start, end in zip(starts[:, 0], ends[:, 0])
                ]
            else:
                return [
                    (
                        pd.Timestamp(labels.index[start]),
                        pd.Timestamp(labels.index[end - 1]),
                    )
                    if start != end - 1
                    else pd.Timestamp(labels.index[start])
                    for start, end in zip(starts[:, 0], ends[:, 0])
                ]
    else:
        return {
            col: to_events(labels[col], freq_as_period, merge_consecutive)
            for col in labels.columns
        }


def to_labels(lists, time_index, freq_as_period=True):
    """Convert event list(s) to binary series along a time line.

    Parameters
    ----------
    lists: list or dict
        A list of events, or a dict of lists of events.
        If list, it represents a single type of event;
        If dict, each key-value pair represents a type of event.
        Each event in a list can be a pandas Timestamp, or a tuple of two
        Timestamps that is regarded as a closed interval.

    time_index: pandas DatatimeIndex
        Time index to build the label series.

    freq_as_period: bool, optional
        Whether to regard time index with regular frequency (i.e. attribute
        `freq` of time index is not None) as time spans.
        E.g. DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03',
        '2017-01-04', '1970-01-05'], dtype='datetime64[ns]', freq='D') has
        daily frequency. If freq_as_period=True, each element represents that
        day, and that day will be marked positive if an event in the event list
        overlaps with any part of that day. Otherwsie, each time element
        represents the instantaneous time stamp 00:00:00 on that day, and that
        time point will be marked positive if an event in the event list covers
        it.
        Default: True.

    Returns
    -------
    pandas Series or DataFrame
        Series of binary labels. If input is asingle list, the output is a
        Series, otherwise if input is a dict, the output is a DataFrame.

    """

    if not isinstance(time_index, pd.DatetimeIndex):
        raise TypeError("Time index must be a pandas DatetimeIndex object.")

    if not time_index.is_monotonic_increasing:
        raise ValueError("Time index must be monotoic increasing.")

    if isinstance(lists, list):
        labels = pd.Series(False, index=time_index)
        lists = validate_events(lists)
        if freq_as_period and (time_index.freq is not None):
            period_end = pd.date_range(
                start=time_index[1],
                periods=len(time_index),
                freq=time_index.freq,
            ) - pd.Timedelta("1ns")
            for event in lists:
                if isinstance(event, tuple):
                    isOverlapped = (time_index <= event[1]) & (
                        period_end >= event[0]
                    )
                else:
                    isOverlapped = (time_index <= event) & (
                        period_end >= event
                    )
                labels.loc[isOverlapped] = True
        else:
            for event in lists:
                if isinstance(event, tuple):
                    labels.loc[
                        (labels.index >= event[0]) & (labels.index <= event[1])
                    ] = True
                else:
                    labels.loc[labels.index == event] = True
    elif isinstance(lists, dict):
        labels = pd.DataFrame(False, index=time_index, columns=lists.keys())
        for col, key in zip(labels.columns, lists.keys()):
            labels[col] = to_labels(lists[key], time_index, freq_as_period)
    else:
        raise TypeError("Argument `lists` must be a list or a dict.")

    return labels


def expand_events(lists, left_expand, right_expand):
    """Expand time windows in an event list.

    Given a list of events, expand the duration of events by a given factor.
    This may help to process true event list before calculating the quality
    of a detection result using a scoring function, if slight offset in result
    is considered acceptable.

    Parameters
    ----------
    lists: list or dict
        A list of events, or a dict of lists of events.
        If dict, each key-value pair represents an independent type of event.
        Each event in a list can be a pandas Timestamp, or a tuple of two
        Timestamps that is regarded as a closed interval.

    left_expand: pandas Timedelta
        Time range to expand backward.

    right_expand: pandas Timedelta
        Time range to expand forward.

    Returns
    -------
    list or dict
        Expanded events.

    """

    if isinstance(lists, list):
        expanded = []
        for ano in lists:
            if isinstance(ano, tuple):
                expanded.append((ano[0] - left_expand, ano[1] + right_expand))
            else:
                expanded.append((ano - left_expand, ano + right_expand))
        expanded = validate_events(expanded)
    elif isinstance(lists, dict):
        expanded = {
            key: expand_events(value, left_expand, right_expand)
            for key, value in lists.items()
        }
    else:
        raise TypeError("Arugment `lists` must be a list or a dict of lists.")

    return expanded


def resample(ts, dT=None):
    """Resample the time points of a time series with given constant spacing.
    The values at new time points are calcuated by time-weighted linear
    interpolation.

    Parameters
    ----------
    ts: pandas Series or DataFrame
        Time series to resample. Index of the object must be DatetimeIndex.

    dT: pandas Timedelta, optional
        The new constant time step. If not given, the greatest common divider
        of original time steps will be used, which makes the refinement a
        minimal refinement subject to keeping all original time points still
        included in the resampled time series. Please note that this may
        dramatically increase the size of time series and memory usage.
        Default: None.

    Returns
    -------
    pandas Series or DataFrame
        Resampled time series.

    """

    def gcd_of_array(arr):
        """Get the GCD of an array of integer"""
        return reduce(gcd, arr)

    ts = validate_series(ts)

    isSeries = False
    if isinstance(ts, pd.Series):
        isSeries = True
        seriesName = ts.name
        ts = ts.to_frame()

    if dT is None:
        dT = pd.Timedelta(
            np.timedelta64(gcd_of_array([int(dt) for dt in np.diff(ts.index)]))
        )

    rdf = pd.DataFrame(index=pd.date_range(ts.index[0], ts.index[-1], freq=dT))

    rdf = rdf.join(ts, how="outer")
    rdf = rdf.interpolate("index")

    rdf = rdf.reindex(pd.date_range(ts.index[0], ts.index[-1], freq=dT))

    if isSeries:
        rdf = rdf[rdf.columns[0]]
        rdf.name = seriesName

    return rdf


def split_train_test(ts, mode=1, n_splits=1, train_ratio=0.7):
    """Split time series into training and testing set for cross validation.

    Parameters
    ----------
    ts: pandas Series or DataFrame
        Time series to process.

    mode: int, optional
        The split mode to use. Choose from 1, 2, 3 and 4.

        1. Divide time series into n_splits folds of equal length, split
           each fold into training and testing based on train_ratio.

        2. Create n_splits folds, where each fold starts at t_0 and ends
           at t_(n/n_splits), where n goes from 0 to n_splits and the
           first train_ratio of the fold is for training.

        3. Create n_splits folds, where each fold starts at t_0. Each fold
           has len(ts)/(1 + n_splits) test points at the end. Each fold is
           n * len(ts)/(1 + n_splits) long, where n ranges from 1 to
           n_splits.

        4. Create n_splits folds, where each fold starts at t_0. Each fold
           has n * len(ts)/(1 + n_splits) training points at the beginning
           of the time series, where n ranges from 1 to n_splits and the
           remaining points are testing points.

        Default: 1.

    n_splits: int, optional
        Number of splits. Default: 1.

    train_ratio: float, optional
        Ratio between length of training series and each fold, only used by
        mode 1 and 2. Default: 0.7.

    Returns
    -------
        tuple (list, list)
            - list of pandas Series or DataFrame: Training time series
            - list of pandas Series or DataFrame: Testing time series

    Examples
    --------
        In the following description of the four modes, 1s represent positions
        assigned to training, 2s represent those assigned to testing, 0s are
        those not assigned.

        For a time series with length 40, if `n_splits=4`, `train_ratio=0.7`,

        - If `mode=1`:

            1111111222000000000000000000000000000000
            0000000000111111122200000000000000000000
            0000000000000000000011111112220000000000
            0000000000000000000000000000001111111222

        - If `mode=2`:

            1111111222000000000000000000000000000000
            1111111111111122222200000000000000000000
            1111111111111111111112222222220000000000
            1111111111111111111111111111222222222222

        - If `mode=3`:

            1111111122222222000000000000000000000000
            1111111111111111222222220000000000000000
            1111111111111111111111112222222200000000
            1111111111111111111111111111111122222222

        - If `mode=4`:

            1111111122222222222222222222222222222222
            1111111111111111222222222222222222222222
            1111111111111111111111112222222222222222
            1111111111111111111111111111111122222222

    """

    if not isinstance(ts, (pd.DataFrame, pd.Series)):
        raise ValueError("Argument `ts` must be a pandas Series or DataFrame.")

    if mode == 1:
        fold_len = round(len(ts) / n_splits)
        ts_train = []
        ts_test = []
        fold_pos = 0
        for _ in range(n_splits - 1):
            ts_train.append(
                ts.iloc[fold_pos : (fold_pos + round(fold_len * train_ratio))]
            )
            ts_test.append(
                ts.iloc[
                    (fold_pos + round(fold_len * train_ratio)) : (
                        fold_pos + fold_len
                    )
                ]
            )
            fold_pos = fold_pos + fold_len
        ts_train.append(
            ts.iloc[
                fold_pos : (
                    fold_pos + round((len(ts) - fold_pos) * train_ratio)
                )
            ]
        )
        ts_test.append(
            ts.iloc[(fold_pos + round((len(ts) - fold_pos) * train_ratio)) :]
        )
    elif mode == 2:
        ts_train = []
        ts_test = []
        for k_fold in range(n_splits - 1):
            fold_len = round(len(ts) / n_splits) * (k_fold + 1)
            ts_train.append(ts.iloc[: round(fold_len * train_ratio)])
            ts_test.append(ts.iloc[round(fold_len * train_ratio) : fold_len])
        ts_train.append(ts.iloc[: round(len(ts) * train_ratio)])
        ts_test.append(ts.iloc[round(len(ts) * train_ratio) : len(ts)])
    elif mode == 3:
        ts_train = []
        ts_test = []
        fold_len = round(len(ts) / (n_splits + 1))
        for k_fold in range(n_splits - 1):
            ts_train.append(ts.iloc[: ((k_fold + 1) * fold_len)])
            ts_test.append(
                ts.iloc[((k_fold + 1) * fold_len) : ((k_fold + 2) * fold_len)]
            )
        ts_train.append(ts.iloc[: (n_splits * fold_len)])
        ts_test.append(ts.iloc[(n_splits * fold_len) :])
    elif mode == 4:
        ts_train = []
        ts_test = []
        fold_len = round(len(ts) / (n_splits + 1))
        for k_fold in range(n_splits):
            ts_train.append(ts.iloc[: ((k_fold + 1) * fold_len)])
            ts_test.append(ts.iloc[((k_fold + 1) * fold_len) :])
    else:
        raise ValueError("Argument `mode` must be one of 1, 2, 3, and 4.")
    # rename series
    if isinstance(ts, pd.Series):
        if ts.name is None:
            ts_train = [
                s.rename("train_{}".format(i)) for i, s in enumerate(ts_train)
            ]
            ts_test = [
                s.rename("test_{}".format(i)) for i, s in enumerate(ts_test)
            ]
        else:
            ts_train = [
                s.rename("{}_train_{}".format(ts.name, i))
                for i, s in enumerate(ts_train)
            ]
            ts_test = [
                s.rename("{}_test_{}".format(ts.name, i))
                for i, s in enumerate(ts_test)
            ]
    else:
        for i, df in enumerate(ts_train):
            ts_train[i].columns = [
                "{}_train_{}".format(col, i) for col in ts_train[i].columns
            ]
        for i, df in enumerate(ts_test):
            ts_test[i].columns = [
                "{}_test_{}".format(col, i) for col in ts_test[i].columns
            ]
    return ts_train, ts_test
