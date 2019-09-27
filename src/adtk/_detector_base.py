import pandas as pd

from ._base import _Model1D, _ModelHD
from .data import to_events
from .metrics import recall, precision, f1_score, iou


class _Detector1D(_Model1D):
    def fit(self, ts):
        """Train the detector with given time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used to train the detector.
            If a DataFrame with k columns, k univariate detectors will be
            trained independently.

        """
        self._fit(ts)

    def detect(self, ts, return_list=False):
        """Detect anomalies from given time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, k univariate detectors will be
            applied to them independently.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series, pandas.DataFrame, list, or dict
            Detected anomalies.
            If input is a Series and return_list=False, return a Series;
            If input is a DataFrame and return_list=False, return a DataFrame,
            where each column corresponds a column in input;
            If input is a Series and return_list=True, return a list of time
            stamps or time stamp tuples;
            If input is a DataFrame and return_list=True, return a dict of
            lists, where each key-value pair corresponds a column in input.

        """
        detected = self._predict(ts)
        if return_list:
            if isinstance(detected, pd.Series):
                return to_events(detected)
            else:
                return {
                    col: to_events(detected[col]) for col in detected.columns
                }
        else:
            return detected

    def fit_detect(self, ts, return_list=False):
        """Train the detector and detect anomalies from the time series used
        for training.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used for training and be detected for anomalies.
            If a DataFrame with k columns, k univariate detectors will be
            trained and applied to them independently.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series, pandas.DataFrame, list, or dict
            Detected anomalies.
            If input is a Series and return_list=False, return a Series;
            If input is a DataFrame and return_list=False, return a DataFrame,
            where each column corresponds a column in input;
            If input is a Series and return_list=True, return a list of time
            stamps or time stamp tuples;
            If input is a DataFrame and return_list=True, return a dict of
            lists, where each key-value pair corresponds a column in input.

        """
        self.fit(ts)
        return self.detect(ts, return_list=return_list)

    def predict(self, ts, return_list=False):
        """
        Alias of `detect`.
        """
        return self.detect(ts, return_list=return_list)

    def fit_predict(self, ts, return_list=False):
        """
        Alias of `fit_detect`.
        """
        return self.fit_detect(ts, return_list=return_list)

    def score(self, ts, anomaly_true, scoring="recall", **kwargs):
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or pandas.DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, k univariate detectors will be
            applied to them independently.

        anomaly_true: pandas.Series, pandas.DataFrame, list, or dict
            True anomalies.
            If Series, it is a series binary labels indicating anomalous;
            If DataFrame, each column is considered as an independent type of
            anomaly;
            If list, it is a list of anomalous events in form of time points
            (pandas.Timestamp) or time windows (2-tuple of time stamps);
            If a dict of lists, each value is considered as an independent type
            of anomaly.

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
            scoring_func = recall
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


class _DetectorHD(_ModelHD):
    def fit(self, df):
        """Train the detector with given time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used to train the detector.

        """
        self._fit(df)

    def detect(self, df, return_list=False):
        """Detect anomalies from given time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to detect anomalies from.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series or list
            Detected anomalies.
            If return_list=False, return a binary series;
            If return_list=True, return a list of time stamps or time stamp
            tuples.

        """
        detected = self._predict(df)
        if return_list:
            return to_events(detected)
        else:
            return detected

    def fit_detect(self, df, return_list=False):
        """Train the detector and detect anomalies from the time series used
        for training.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used for training and be detected for anomalies.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas.Series or list
            Detected anomalies.
            If return_list=False, return a binary series;
            If return_list=True, return a list of time stamps or time stamp
            tuples.

        """
        self.fit(df)
        return self.detect(df, return_list=return_list)

    def predict(self, df, return_list=False):
        """
        Alias of `detect`.
        """
        return self.detect(df, return_list=return_list)

    def fit_predict(self, df, return_list=False):
        """
        Alias of `fit_detect`.
        """
        return self.fit_detect(df, return_list=return_list)

    def score(self, df, anomaly_true, scoring="recall", **kwargs):
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        df: pandas DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, k univariate detectors will be
            applied to them independently.

        anomaly_true: Series, or a list of Timestamps or Timestamp tuple
            True anomalies.
            If Series, it is a series binary labels indicating anomalous;
            If list, it is a list of anomalous events in form of time windows.

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
            scoring_func = recall
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
