"""Module for aggregators.

An aggregator combines multiple lists of anomalies into one.

"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd

from .._aggregator_base import _Aggregator
from ..data import validate_events


class CustomizedAggregator(_Aggregator):
    """Aggregator derived from a user-given function and parameters.

    Parameters
    ----------
    aggregate_func: function
        A function aggregating multiple types of anomaly.

        The first input argument must be a pandas DataFrame, a dict of pandas
        Series/DataFrame, or a dict of event lists.

        - If a pandas DataFrame, every column is a binary Series representing a
          type of anomaly.
        - If a dict of pandas Series/DataFrame, every value of the dict is a
          binary Series/DataFrame representing a type or some types of anomaly;
        - If a dict of list, every value of the dict is a type of anomaly as a
          list of events, where each event is represented as a pandas Timestamp
          if it is instantaneous or a 2-tuple of pandas Timestamps if it is a
          closed time interval.

        Optional input argument may be accepted through parameter
        `aggregate_func_params`.

        The output must be a list of pandas Timestamps.

        - If input is a pandas DataFrame or a dict of Series/DataFrame, return
          a single binary pandas Series;
        - If input is a dict of lists, return a single list of events.

    aggregate_func_params: dict, optional
        Parameters of `aggregate_func`. Default: None.

    """

    def __init__(
        self,
        aggregate_func: Callable,
        aggregate_func_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.aggregate_func = aggregate_func
        self.aggregate_func_params = aggregate_func_params

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("aggregate_func", "aggregate_func_params")

    def _predict_core(
        self,
        lists: Union[
            pd.DataFrame,
            Dict[str, Union[pd.Series, pd.DataFrame]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
        if self.aggregate_func_params is None:
            aggregate_func_params = {}
        else:
            aggregate_func_params = self.aggregate_func_params
        return self.aggregate_func(lists, **aggregate_func_params)


class OrAggregator(_Aggregator):
    """Aggregator that identifies a time point as anomalous as long as it is
    included in one of the input anomaly lists.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return tuple()

    def _predict_core(
        self,
        lists: Union[
            pd.DataFrame,
            Dict[str, Union[pd.Series, pd.DataFrame]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
        if isinstance(lists, dict):
            if isinstance(next(iter(lists.values())), list):
                clean_lists = {
                    key: validate_events(value) for key, value in lists.items()
                }
                return validate_events(
                    [
                        window
                        for clean_predict in clean_lists.values()
                        for window in clean_predict
                    ]
                )
            else:  # a dict of pandas Series/DataFrame
                return self._predict_core(
                    pd.concat(lists, join="outer", axis=1)
                )
        else:  # pandas DataFrame
            predicted = lists.any(axis=1)
            predicted[~predicted & lists.isna().any(axis=1)] = float("nan")
            return predicted


class AndAggregator(_Aggregator):
    """Aggregator that identifies a time point as anomalous only if it is
    included in all the input anomaly lists.
    """

    def __init__(self) -> None:
        super().__init__()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return tuple()

    def _predict_core(
        self,
        lists: Union[
            pd.DataFrame,
            Dict[str, Union[pd.Series, pd.DataFrame]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
        if isinstance(lists, dict):
            if isinstance(next(iter(lists.values())), list):
                clean_lists = {
                    key: validate_events(value, point_as_interval=True)
                    for key, value in lists.items()
                }
                time_window_stats = {
                    key: pd.Series(
                        [0] * len(clean_predict)
                        + [1] * 2 * len(clean_predict)
                        + [0] * len(clean_predict),
                        index=(
                            [
                                window[0] - pd.Timedelta("1ns")
                                for window in clean_predict
                            ]
                            + [window[0] for window in clean_predict]
                            + [window[1] for window in clean_predict]
                            + [
                                window[1] + pd.Timedelta("1ns")
                                for window in clean_predict
                            ]
                        ),
                        dtype=int,
                    ).sort_index()
                    for key, clean_predict in clean_lists.items()
                }  # type: Union[Dict, pd.Series]
                time_window_stats = {
                    key: value[~value.index.duplicated()]
                    for key, value in time_window_stats.items()
                }
                time_window_stats = (
                    pd.concat(time_window_stats, axis=1, join="outer")
                    .fillna(method="ffill")
                    .fillna(method="bfill")
                    .fillna(0)
                )
                time_window_stats = time_window_stats.all(axis=1)
                status = 0
                last_t = None
                aggregated_predict = []
                for t, v in time_window_stats.items():
                    if (status == 0) and (v == 1):
                        start = t
                        status = 1
                    if (status == 1) and (v == 0):
                        end = last_t
                        aggregated_predict.append((start, end))
                        status = 0
                    last_t = t
                return validate_events(aggregated_predict)
            else:  # a dict of pandas Series/DataFrame
                return self._predict_core(
                    pd.concat(lists, join="outer", axis=1)
                )
        else:  # pandas DataFrame
            predicted = lists.all(axis=1)
            predicted[predicted & lists.isna().any(axis=1)] = float("nan")
            return predicted
