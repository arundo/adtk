"""Module for aggregators.

An aggregator combines multiple lists of anomalies into one.

"""

import pandas as pd

from .._aggregator_base import _Aggregator
from ..data import validate_events

__all__ = ["OrAggregator", "AndAggregator", "CustomizedAggregator"]


class CustomizedAggregator(_Aggregator):
    """Aggregator derived from a user-given function and parameters.

    Parameters
    ----------
    aggregate_func: function
        A function aggregating mulitple lists of anomalies. The first input
        argument must be a dict, optional input argument allows (through
        parameter `aggregate_func_params`). The output must be a list of pandas
        Timestamps.

    aggregate_func_params: dict, optional
        Parameters of aggregate_func. Default: None.

    """

    _default_params = {
        "aggregate_func": (lambda lists: []),
        "aggregate_func_params": None,
    }

    def __init__(
        self,
        aggregate_func=_default_params["aggregate_func"],
        aggregate_func_params=_default_params["aggregate_func_params"],
    ):
        super().__init__(
            aggregate_func=aggregate_func,
            aggregate_func_params=aggregate_func_params,
        )

    def _predict_core(self, lists):
        if self.aggregate_func_params is None:
            aggregate_func_params = {}
        else:
            aggregate_func_params = self.aggregate_func_params
        return self.aggregate_func(lists, **aggregate_func_params)


class OrAggregator(_Aggregator):
    """Aggregator that identifies a time point as anomalous as long as it is
    included in one of the input anomaly lists.
    """

    _need_fit = False

    def __init__(self):
        super().__init__()

    def _predict_core(self, lists):
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

    _need_fit = False

    def __init__(self):
        super().__init__()

    def _predict_core(self, lists):
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
                    ).sort_index()
                    for key, clean_predict in clean_lists.items()
                }
                time_window_stats = {
                    key: value[~value.index.duplicated()]
                    for key, value in time_window_stats.items()
                }
                time_window_stats = (
                    pd.concat(time_window_stats, axis=1, join="outer")
                    .fillna(method="ffill")
                    .fillna(method="bfill")
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
