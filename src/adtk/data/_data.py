"""Module is for data (time series and anomaly list) processing.
"""

from typing import Dict, List, Optional, Tuple, Union, overload

import numpy as np
import pandas as pd


def validate_series(
    ts: Union[pd.Series, pd.DataFrame],
    check_freq: bool = True,
    check_categorical: bool = False,
) -> Union[pd.Series, pd.DataFrame]:
    """Validate time series.

    This functoin will check some common critical issues of time series that
    may cause problems if anomaly detection is performed without fixing them.
    The function will automatically fix some of them and raise errors for the
    others.

    Issues will be checked and automatically fixed include:

    - Time index is not monotonically increasing;
    - Time index contains duplicated time stamps (fix by keeping first values);
    - (optional) Time index attribute `freq` is missed while the index follows
      a frequency;
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
        if (ts.index.freq is None) and (ts.index.inferred_freq is not None):
            ts = ts.asfreq(ts.index.inferred_freq)

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


def validate_events(
    event_list: List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
    point_as_interval: bool = False,
) -> List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]:
    """Validate event list.

    This function will check and fix some common issues in an event list (a
    list of time windows), including invalid time window, overlapped time
    windows, unsorted events, etc.

    Parameters
    ----------
    event_list: list
        A list of events, where an event is a pandas Timestamp if it is
        instantaneous or a 2-tuple of pandas Timestamps if it is a closed time
        interval.

    point_as_interval: bool, optional
        Whether to return all instantaneous event as a close interval with
        identicial start point and end point. Default: False.

    Returns
    -------
    list:
        A validated list of events.

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

    time_window_ends = []  # type: List[pd.Timestamp]
    time_window_type = []  # type: List[int]
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
        time_window_type, index=pd.DatetimeIndex(time_window_ends), dtype=int
    )  # type: pd.Series
    time_window_end_series.sort_index(kind="mergesort", inplace=True)
    time_window_end_series = time_window_end_series.cumsum()
    status = 0
    merged_event_list = (
        []
    )  # type: List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    for t, v in time_window_end_series.iteritems():  # type: pd.Timestamp, int
        if (status == 0) and (v > 0):
            start = t  # type: pd.Timestamp
            status = 1
        if (status == 1) and (v <= 0):
            end = t  # type: pd.Timestamp
            merged_event_list.append([start, end])
            status = 0
    for i in range(1, len(merged_event_list)):
        this_start = merged_event_list[i][0]  # type: pd.Timestamp
        this_end = merged_event_list[i][1]  # type: pd.Timestamp
        last_start = merged_event_list[i - 1][0]  # type: pd.Timestamp
        last_end = merged_event_list[i - 1][1]  # type: pd.Timestamp
        if last_end + pd.Timedelta("1ns") >= this_start:
            merged_event_list[i] = [last_start, this_end]
            merged_event_list[i - 1] = None
    merged_event_list = [
        w[0] if (w[0] == w[1] and not point_as_interval) else tuple(w)
        for w in merged_event_list
        if w is not None
    ]
    return merged_event_list


@overload
def to_events(
    labels: pd.Series,
    freq_as_period: bool = True,
    merge_consecutive: Optional[bool] = None,
) -> List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]:
    ...


@overload
def to_events(  # type: ignore
    labels: pd.DataFrame,
    freq_as_period: bool = True,
    merge_consecutive: Optional[bool] = None,
) -> Dict[str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]]:
    ...


def to_events(
    labels: Union[pd.Series, pd.DataFrame],
    freq_as_period: bool = True,
    merge_consecutive: Optional[bool] = None,
) -> Union[
    List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
    Dict[str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]],
]:
    """Convert binary label series to event list.

    Parameters
    ----------
    labels: pandas Series or DataFrame
        Binary series of anomaly labels. If a DataFrame, each column is
        regarded as a type of anomaly independently.

    freq_as_period: bool, optional
        Whether to regard time index with regular frequency (i.e. attribute
        `freq` of time index is not None) as time intervals.

        For example, DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03',
        '2017-01-04', '2017-01-05'], dtype='datetime64[ns]', freq='D') has
        daily frequency. If freq_as_period=True, each time point in the index
        represents that day (24 hours). Otherwsie, each time point represents
        the instantaneous time instance of 00:00:00 on that day.

        Default: True.

    merge_consecutive: bool, optional
        Whether to merge consecutive events into a single time window. If not
        specified, it is on automatically if the input time index has a regular
        frequency and freq_as_period=True, and it is off otherwise. Default:
        None.

    Returns
    -------
    list or dict
        - If input is a Series, output is a list of events where an event is a
          pandas Timestamp if it is instantaneous or a 2-tuple of pandas
          Timestamps if it is a closed time interval.
        - If input is a DataFrame, every column is treated as an independent
          binary series, and output is a dict where keys are column names and
          values are event lists.

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
                ) - pd.Timedelta(
                    "1ns"
                )  # type: pd.DatetimeIndex
                return [
                    (start, end) if start != end else start
                    for start, end in zip(
                        list(labels.index[labels]), list(period_end[labels])
                    )
                ]
            else:
                return list(labels.index[labels])
        else:
            labels_values = labels.values.astype(int).reshape(
                -1, 1
            )  # type: np.ndarray
            mydiff = np.vstack(
                [
                    labels_values[0, :] - 0,
                    np.diff(labels_values, axis=0),
                    0 - labels_values[-1, :],
                ]
            )  # type: np.ndarray
            starts = np.argwhere(mydiff == 1)  # type: np.ndarray
            ends = np.argwhere(mydiff == -1)  # type: np.ndarray
            if freq_as_period and (labels.index.freq is not None):
                period_end = pd.date_range(
                    start=labels.index[1],
                    periods=len(labels.index),
                    freq=labels.index.freq,
                ) - pd.Timedelta("1ns")
                return [
                    (labels.index[start], period_end[end - 1])
                    if labels.index[start] != period_end[end - 1]
                    else labels.index[start]
                    for start, end in zip(starts[:, 0], ends[:, 0])
                ]
            else:
                return [
                    (labels.index[start], labels.index[end - 1])
                    if start != end - 1
                    else labels.index[start]
                    for start, end in zip(starts[:, 0], ends[:, 0])
                ]
    else:
        if labels.columns.duplicated().any():
            raise ValueError("Input DataFrame must have unique column names.")
        return {
            col: to_events(labels[col], freq_as_period, merge_consecutive)
            for col in labels.columns
        }


@overload
def to_labels(
    lists: List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
    time_index: pd.DatetimeIndex,
    freq_as_period: bool = True,
) -> pd.Series:
    ...


@overload
def to_labels(
    lists: Dict[
        str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ],
    time_index: pd.DatetimeIndex,
    freq_as_period: bool = True,
) -> pd.DataFrame:
    ...


def to_labels(
    lists: Union[
        List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
        Dict[
            str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        ],
    ],
    time_index: pd.DatetimeIndex,
    freq_as_period: bool = True,
) -> Union[pd.Series, pd.DataFrame]:
    """Convert event list to binary series along a time index.

    Parameters
    ----------
    lists: list or dict
        A list of events, or a dict of lists of events.

        - If list, a list of events where an event is a pandas Timestamp if it
          is instantaneous or a 2-tuple of pandas Timestamps if it is a closed
          time interval.
        - If dict, each key-value pair represents an independent list of
          events.

    time_index: pandas DatatimeIndex
        Time index to build the label series.

    freq_as_period: bool, optional
        Whether to regard time index with regular frequency (i.e. attribute
        `freq` of time index is not None) as time intervals.

        For example, DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03',
        '2017-01-04', '2017-01-05'], dtype='datetime64[ns]', freq='D') has
        daily frequency. If freq_as_period=True, each time piont represents
        that day, and that day will be marked positive if an event in the event
        list overlaps with the period of that day (24 hours). Otherwsie, each
        time point represents the instantaneous time instance of 00:00:00 on
        that day, and that time point will be marked positive if an event in
        the event list covers it.

        Default: True.

    Returns
    -------
    pandas Series or DataFrame
        Series of binary labels.

        - If input is asingle list, the output is a Series.
        - If input is a dict of lists, the output is a DataFrame where each
          column corresponds a list in the dict.

    """

    if not isinstance(time_index, pd.DatetimeIndex):
        raise TypeError("Time index must be a pandas DatetimeIndex object.")

    if not time_index.is_monotonic_increasing:
        raise ValueError("Time index must be monotoic increasing.")

    if isinstance(lists, list):
        labels = pd.Series(False, index=time_index)  # type: pd.Series
        lists = validate_events(lists)
        if freq_as_period and (time_index.freq is not None):
            period_end = pd.date_range(
                start=time_index[1],
                periods=len(time_index),
                freq=time_index.freq,
            ) - pd.Timedelta(
                "1ns"
            )  # type: pd.DatetimeIndex
            for event in lists:
                isOverlapped = pd.Series(
                    index=time_index, dtype=bool
                )  # type: pd.Series
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
        return labels
    elif isinstance(lists, dict):
        labels_df = pd.DataFrame(
            False, index=time_index, columns=lists.keys()
        )  # pd.DataFrame
        for col, key in zip(labels_df.columns, lists.keys()):
            labels_df[col] = to_labels(lists[key], time_index, freq_as_period)
        return labels_df
    else:
        raise TypeError("Argument `lists` must be a list or a dict.")


@overload
def expand_events(
    lists: List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
    left_expand: Union[pd.Timedelta, str, int] = 0,
    right_expand: Union[pd.Timedelta, str, int] = 0,
    freq_as_period: bool = True,
) -> List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]:
    ...


@overload
def expand_events(
    lists: Dict[
        str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ],
    left_expand: Union[pd.Timedelta, str, int] = 0,
    right_expand: Union[pd.Timedelta, str, int] = 0,
    freq_as_period: bool = True,
) -> Dict[str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]]:
    ...


@overload
def expand_events(
    lists: pd.Series,
    left_expand: Union[pd.Timedelta, str, int] = 0,
    right_expand: Union[pd.Timedelta, str, int] = 0,
    freq_as_period: bool = True,
) -> pd.Series:
    ...


@overload
def expand_events(  # type:ignore
    lists: pd.DataFrame,
    left_expand: Union[pd.Timedelta, str, int] = 0,
    right_expand: Union[pd.Timedelta, str, int] = 0,
    freq_as_period: bool = True,
) -> pd.DataFrame:
    ...


def expand_events(  # type:ignore
    events: Union[
        List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
        Dict[
            str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        ],
        pd.Series,
        pd.DataFrame,
    ],
    left_expand: Union[pd.Timedelta, str, int] = 0,
    right_expand: Union[pd.Timedelta, str, int] = 0,
    freq_as_period: bool = True,
) -> Union[
    List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
    Dict[str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]],
    pd.Series,
    pd.DataFrame,
]:
    """Expand duration of events.

    Parameters
    ----------
    events: list, dict, pandas Series, or pandas DataFrame
        Events to be expanded.

        - If list, a list of events where an event is a pandas Timestamp if it
          is instantaneous or a 2-tuple of pandas Timestamps if it is a closed
          time interval.
        - If dict, each key-value pair represents an independent list of
          events.
        - If pandas Series, it is binary where 1 represents events cover this
          time point.
        - If pandas DataFrame, each column is treated as an independent Series.

    left_expand: pandas Timedelta, str, or int, optional
        Time range to expand backward.

        - If str, it must be able to be converted into a pandas Timedelta
          object.
        - If int, it must be in nanosecond.

        Default: 0.

    right_expand: pandas Timedelta, str, or int, optional
        Time range to expand forward.

        - If str, it must be able to be converted into a pandas Timedelta
          object.
        - If int, it must be in nanosecond.

        Default: 0.

    freq_as_period: bool, optional
        Whether to regard time index with regular frequency (i.e. attribute
        `freq` of time index is not None) as time intervals. Only used when
        input events is pandas Series or DataFrame.

        For example, DatetimeIndex(['2017-01-01', '2017-01-02', '2017-01-03',
        '2017-01-04', '2017-01-05'], dtype='datetime64[ns]', freq='D') has
        daily frequency. If freq_as_period=True, each time point in the index
        represents that day (24 hours). Otherwsie, each time point represents
        the instantaneous time instance of 00:00:00 on that day.

        Default: True.

    Returns
    -------
    list, dict, pandas Series, or pandas DataFrame
        Expanded events.

    """

    if not isinstance(left_expand, pd.Timedelta):
        left_expand = pd.Timedelta(left_expand)
    if not isinstance(right_expand, pd.Timedelta):
        right_expand = pd.Timedelta(right_expand)

    if isinstance(events, pd.Series):
        labels = validate_series(events)  # type: pd.Series
        lists = to_events(
            labels, freq_as_period=freq_as_period
        )  # type:List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        expanded_lists = expand_events(  # type:ignore
            events=lists, left_expand=left_expand, right_expand=right_expand
        )  # type:List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        expanded_labels = to_labels(
            lists=expanded_lists,
            time_index=labels.index,
            freq_as_period=freq_as_period,
        )  # type: pd.Series
        expanded_labels.loc[
            (expanded_labels == False) & (labels.isna())
        ] = float("nan")
        expanded_labels = expanded_labels.rename(labels.name)
        expanded_labels.index = labels.index
        return expanded_labels
    elif isinstance(events, pd.DataFrame):
        expanded_df = pd.concat(
            [
                expand_events(
                    s,
                    left_expand=left_expand,
                    right_expand=right_expand,
                    freq_as_period=freq_as_period,
                )
                for _, s in events.iteritems()
            ],
            axis=1,
        )  # type: pd.DataFrame
        return expanded_df
    elif isinstance(events, list):
        expanded_list = (
            []
        )  # type: List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        for ano in events:
            if isinstance(ano, tuple):
                expanded_list.append(
                    (ano[0] - left_expand, ano[1] + right_expand)
                )
            else:
                expanded_list.append((ano - left_expand, ano + right_expand))
        expanded_list = validate_events(expanded_list)
        return expanded_list
    elif isinstance(events, dict):
        return {
            key: expand_events(value, left_expand, right_expand)
            for key, value in events.items()
        }
    else:
        raise TypeError("Arugment `events` must be a list or a dict of lists.")


def split_train_test(
    ts: Union[pd.Series, pd.DataFrame],
    mode: int = 1,
    n_splits: int = 1,
    train_ratio: float = 0.7,
) -> List[
    Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]
]:
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
        list of 2-tuples (train, test)
            Splitted training and testing series.

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

    splits = []
    if mode == 1:
        fold_len = round(len(ts) / n_splits)
        fold_pos = 0
        for _ in range(n_splits - 1):
            splits.append(
                (
                    ts.iloc[
                        fold_pos : (fold_pos + round(fold_len * train_ratio))
                    ],
                    ts.iloc[
                        (fold_pos + round(fold_len * train_ratio)) : (
                            fold_pos + fold_len
                        )
                    ],
                )
            )
            fold_pos = fold_pos + fold_len
        splits.append(
            (
                ts.iloc[
                    fold_pos : (
                        fold_pos + round((len(ts) - fold_pos) * train_ratio)
                    )
                ],
                ts.iloc[
                    (fold_pos + round((len(ts) - fold_pos) * train_ratio)) :
                ],
            )
        )
    elif mode == 2:
        for k_fold in range(n_splits - 1):
            fold_len = round(len(ts) / n_splits) * (k_fold + 1)
            splits.append(
                (
                    ts.iloc[: round(fold_len * train_ratio)],
                    ts.iloc[round(fold_len * train_ratio) : fold_len],
                )
            )
        splits.append(
            (
                ts.iloc[: round(len(ts) * train_ratio)],
                ts.iloc[round(len(ts) * train_ratio) : len(ts)],
            )
        )
    elif mode == 3:
        fold_len = round(len(ts) / (n_splits + 1))
        for k_fold in range(n_splits - 1):
            splits.append(
                (
                    ts.iloc[: ((k_fold + 1) * fold_len)],
                    ts.iloc[
                        ((k_fold + 1) * fold_len) : ((k_fold + 2) * fold_len)
                    ],
                )
            )
        splits.append(
            (
                ts.iloc[: (n_splits * fold_len)],
                ts.iloc[(n_splits * fold_len) :],
            )
        )
    elif mode == 4:
        fold_len = round(len(ts) / (n_splits + 1))
        for k_fold in range(n_splits):
            splits.append(
                (
                    ts.iloc[: ((k_fold + 1) * fold_len)],
                    ts.iloc[((k_fold + 1) * fold_len) :],
                )
            )
    else:
        raise ValueError("Argument `mode` must be one of 1, 2, 3, and 4.")

    return splits
