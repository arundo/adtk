from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd

from ._base import (
    _NonTrainableUnivariateModel,
    _TrainableMultivariateModel,
    _TrainableUnivariateModel,
)
from .data import to_events
from .metrics import f1_score, iou, precision, recall


class _NonTrainableUnivariateDetector(_NonTrainableUnivariateModel):
    def predict(
        self, ts: Union[pd.Series, pd.DataFrame], return_list: bool = False
    ) -> Union[
        pd.Series,
        pd.DataFrame,
        List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
        Dict[
            str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        ],
    ]:
        """Detect anomalies from given time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to detect anomalies from. If a DataFrame with k
            columns, it is treated as k independent univariate time series, and
            the detector will be applied to each univariate series
            independently.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series, pandas.DataFrame, list, or dict
            Detected anomalies.

            - If input is a Series and return_list=False, return a Series;
            - If input is a DataFrame and return_list=False, return a
              DataFrame, where each column corresponds a column in input;
            - If input is a Series and return_list=True, return a list of time
              stamps or time stamp tuples;
            - If input is a DataFrame and return_list=True, return a dict of
              lists, where each key-value pair corresponds a column in input.

        """
        detected = self._predict(ts)
        if return_list:
            return to_events(detected)
        else:
            return detected

    detect = predict

    def score(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        anomaly_true: Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        scoring: str = "recall",
        **kwargs: Any
    ) -> Union[float, Dict[str, float]]:
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or pandas.DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, it is treated as k independent
            univariate time series, and the detector will be applied to each
            series independently.

        anomaly_true: pandas.Series, pandas.DataFrame, list, or dict
            True anomalies.

            - If pandas Series, it is treated as a series of binary labels.
            - If pandas DataFrame, each column is a binary series and is
              treated as an independent type of anomaly.
            - If list, a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.
            - If dict, each key-value pair is a list of events and is treated
              as an independent type of anomaly.

        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou". See module `metrics` for more information.
            Default: "recall"

        **kwargs
            Optional parameters for scoring function. See module `metrics` for
            more information.

        Returns
        -------
        float or dict
            Score(s) for each type of anomaly.

        """
        if scoring == "recall":
            scoring_func = recall  # type: Callable
        elif scoring == "precision":
            scoring_func = precision
        elif scoring == "f1":
            scoring_func = f1_score
        elif scoring == "iou":
            scoring_func = iou
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'iou'."
            )
        if isinstance(anomaly_true, (pd.Series, pd.DataFrame)):
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=False),
                **kwargs
            )
        else:
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=True),
                **kwargs
            )


class _TrainableUnivariateDetector(_TrainableUnivariateModel):
    def fit(self, ts: Union[pd.Series, pd.DataFrame]) -> None:
        """Train the detector with given time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used to train the detector.
            If a DataFrame with k columns, k univariate detectors will be
            trained independently.

        """
        self._fit(ts)

    def predict(
        self, ts: Union[pd.Series, pd.DataFrame], return_list: bool = False
    ) -> Union[
        pd.Series,
        pd.DataFrame,
        List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
        Dict[
            str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        ],
    ]:
        """Detect anomalies from given time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to detect anomalies from. If a DataFrame with k
            columns, it is treated as k independent univariate time series.

            - If the detector was trained with a Series, the detector will be
              applied to each univariate series independently;
            - If the detector was trained with a DataFrame, i.e. the detector
              is essentially k detectors, those detectors will be applied to
              each univariate series respectively.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series, pandas.DataFrame, list, or dict
            Detected anomalies.

            - If input is a Series and return_list=False, return a Series;
            - If input is a DataFrame and return_list=False, return a
              DataFrame, where each column corresponds a column in input;
            - If input is a Series and return_list=True, return a list of
              events where an event is a pandas Timestamp if it is
              instantaneous or a 2-tuple of pandas Timestamps if it is a closed
              time interval.
            - If input is a DataFrame and return_list=True, return a dict of
              event lists, where each key-value pair corresponds a column in
              input.

        """
        detected = self._predict(ts)
        if return_list:
            return to_events(detected)
        else:
            return detected

    def fit_predict(
        self, ts: Union[pd.Series, pd.DataFrame], return_list: bool = False
    ) -> Union[
        pd.Series,
        pd.DataFrame,
        List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
        Dict[
            str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
        ],
    ]:
        """Train the detector and detect anomalies from the time series used
        for training.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used for training and be detected for anomalies.
            If a DataFrame with k columns, it is treated as k independent
            univariate time series, and k univariate detectors will be trained
            and applied to each series independently.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series, pandas.DataFrame, list, or dict
            Detected anomalies.

            - If input is a Series and return_list=False, return a Series;
            - If input is a DataFrame and return_list=False, return a
              DataFrame, where each column corresponds a column in input;
            - If input is a Series and return_list=True, return a list of
              events where an event is a pandas Timestamp if it is
              instantaneous or a 2-tuple of pandas Timestamps if it is a closed
              time interval.
            - If input is a DataFrame and return_list=True, return a dict of
              event lists, where each key-value pair corresponds a column in
              input.

        """
        self.fit(ts)
        return self.detect(ts, return_list=return_list)

    detect = predict
    fit_detect = fit_predict

    def score(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        anomaly_true: Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        scoring: str = "recall",
        **kwargs: Any
    ) -> Union[float, Dict[str, float]]:
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or pandas.DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, it is treated as k independent
            univariate time series, and k univariate detectors will be applied
            to each series independently.

        anomaly_true: pandas.Series, pandas.DataFrame, list, or dict
            True anomalies.

            - If pandas Series, it is treated as a series of binary labels.
            - If pandas DataFrame, each column is a binary series and is
              treated as an independent type of anomaly.
            - If list, a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.
            - If dict, each key-value pair is a list of events and is treated
              as an independent type of anomaly.

        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou". See module `metrics` for more information.
            Default: "recall"

        **kwargs
            Optional parameters for scoring function. See module `metrics` for
            more information.

        Returns
        -------
        float or dict
            Score(s) for each type of anomaly.

        """
        if scoring == "recall":
            scoring_func = recall  # type: Callable
        elif scoring == "precision":
            scoring_func = precision
        elif scoring == "f1":
            scoring_func = f1_score
        elif scoring == "iou":
            scoring_func = iou
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'iou'."
            )
        if isinstance(anomaly_true, (pd.Series, pd.DataFrame)):
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=False),
                **kwargs
            )
        else:
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=True),
                **kwargs
            )


# class _NonTrainableMultivariateDetector(_NonTrainableMultivariateModel):
#     def detect(
#         self, df: pd.DataFrame, return_list: bool = False
#     ) -> Union[
#         pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
#     ]:
#         """Detect anomalies from given time series.

#         Parameters
#         ----------
#         df: pandas.DataFrame
#             Time series to detect anomalies from.

#         return_list: bool, optional
#             Whether to return a list of anomalous time stamps, or a binary
#             series indicating normal/anomalous. Default: False.

#         Returns
#         -------
#         pandas.Series or list
#             Detected anomalies.

#             - If return_list=False, return a binary series;
#             - If return_list=True, return a list of time stamps or time stamp
#               2-tuples.

#         """
#         detected = self._predict(df)
#         if return_list:
#             return to_events(detected)
#         else:
#             return detected

#     def predict(
#         self, df: pd.DataFrame, return_list: bool = False
#     ) -> Union[
#         pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
#     ]:
#         """
#         Alias of `detect`.
#         """
#         return self.detect(df, return_list=return_list)

#     def score(
#         self,
#         df: pd.DataFrame,
#         anomaly_true: Union[pd.Series, List, Tuple],
#         scoring: str = "recall",
#         **kwargs: Any
#     ) -> float:
#         """Detect anomalies and score the results against true anomalies.

#         Parameters
#         ----------
#         df: pandas DataFrame
#             Time series to detect anomalies from.
#             If a DataFrame with k columns, k univariate detectors will be
#             applied to them independently.

#         anomaly_true: Series, or a list of Timestamps or Timestamp tuple
#             True anomalies.

#             - If Series, it is a series binary labels indicating anomalous;
#             - If list, it is a list of anomalous events in form of time windows.

#         scoring: str, optional
#             Scoring function to use. Must be one of "recall", "precision",
#             "f1", and "iou". See module `metrics` for more information.
#             Default: "recall"

#         **kwargs
#             Optional parameters for scoring function. See module `metrics` for
#             more information.

#         Returns
#         -------
#         float
#             Score of detection result.

#         """
#         if scoring == "recall":
#             scoring_func = recall  # type: Callable
#         elif scoring == "precision":
#             scoring_func = precision
#         elif scoring == "f1":
#             scoring_func = f1_score
#         elif scoring == "iou":
#             scoring_func = iou
#         else:
#             raise ValueError(
#                 "Argument `scoring` must be one of 'recall', 'precision', "
#                 "'f1' and 'iou'."
#             )
#         if isinstance(anomaly_true, pd.Series):
#             return scoring_func(
#                 y_true=anomaly_true,
#                 y_pred=self.detect(df, return_list=False),
#                 **kwargs
#             )
#         else:
#             return scoring_func(
#                 y_true=anomaly_true,
#                 y_pred=self.detect(df, return_list=True),
#                 **kwargs
#             )


class _TrainableMultivariateDetector(_TrainableMultivariateModel):
    def fit(self, df: pd.DataFrame) -> None:
        """Train the detector with given time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used to train the detector.

        """
        self._fit(df)

    def predict(
        self, df: pd.DataFrame, return_list: bool = False
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
        """Detect anomalies from given time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to detect anomalies from.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series or list
            Detected anomalies.

            - If return_list=False, return a binary series;
            - If return_list=True, return a list of events where an event is a
              pandas Timestamp if it is instantaneous or a 2-tuple of pandas
              Timestamps if it is a closed time interval.

        """
        detected = self._predict(df)
        if return_list:
            return to_events(detected)
        else:
            return detected

    def fit_predict(
        self, df: pd.DataFrame, return_list: bool = False
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
        """Train the detector and detect anomalies from the time series used
        for training.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used for training and be detected for anomalies.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series or list
            Detected anomalies.

            - If return_list=False, return a binary series;
            - If return_list=True, return a list of events where an event is a
              pandas Timestamp if it is instantaneous or a 2-tuple of pandas
              Timestamps if it is a closed time interval.

        """
        self.fit(df)
        return self.detect(df, return_list=return_list)

    detect = predict
    fit_detect = fit_predict

    def score(
        self,
        df: pd.DataFrame,
        anomaly_true: Union[pd.Series, List, Tuple],
        scoring: str = "recall",
        **kwargs: Any
    ) -> float:
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        df: pandas DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, k univariate detectors will be
            applied to them independently.

        anomaly_true: Series or list
            True anomalies.

            - If pandas Series, it is treated as a series of binary labels.
            - If list, a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.

        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou". See module `metrics` for more information.
            Default: "recall"

        **kwargs
            Optional parameters for scoring function. See module `metrics` for
            more information.

        Returns
        -------
        float
            Score of detection result.

        """
        if scoring == "recall":
            scoring_func = recall  # type: Callable
        elif scoring == "precision":
            scoring_func = precision
        elif scoring == "f1":
            scoring_func = f1_score
        elif scoring == "iou":
            scoring_func = iou
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'iou'."
            )
        if isinstance(anomaly_true, pd.Series):
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(df, return_list=False),
                **kwargs
            )
        else:
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(df, return_list=True),
                **kwargs
            )
