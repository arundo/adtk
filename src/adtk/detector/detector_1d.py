"""Module for 1-dimensional detectors.

1-dimensional detectors detect anomalies from 1-dimensional time series, i.e.
from pandas Series.

"""

import numpy as np
import pandas as pd
from scipy.stats import t

from sklearn.linear_model import LinearRegression

from ..aggregator import AndAggregator
from .._detector_base import _Detector1D
from ..pipe import Pipenet
from ..transformer import (
    CustomizedTransformer1D,
    DoubleRollingAggregate,
    NaiveSeasonalDecomposition,
    RegressionResidual,
    Retrospect,
    STLDecomposition,
)

__all__ = [
    "ThresholdAD",
    "QuantileAD",
    "InterQuartileRangeAD",
    "GeneralizedESDTestAD",
    "PersistAD",
    "LevelShiftAD",
    "VolatilityShiftAD",
    "AutoregressionAD",
    "SeasonalAD",
    "CustomizedDetector1D",
]


class CustomizedDetector1D(_Detector1D):
    """Detector derived from a user-given function and parameters.

    Parameters
    ----------
    detect_func: function
        A function detecting anomalies from given time series. The first input
        argument must be a pandas Series, optional input argument allows; the
        output must be a binary pandas Series with the same index as input.

    detect_func_params: dict, optional
        Parameters of detect_func. Default: None.

    fit_func: function, optional
        A function learning from a list of time series and return parameters
        dict that detect_func can used for future detection. Default: None.

    fit_func_params: dict, optional
        Parameters of fit_func. Default: None.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _need_fit = False
    _default_params = {
        "detect_func": None,
        "detect_func_params": None,
        "fit_func": None,
        "fit_func_params": None,
    }

    def __init__(
        self,
        detect_func=_default_params["detect_func"],
        detect_func_params=_default_params["detect_func_params"],
        fit_func=_default_params["fit_func"],
        fit_func_params=_default_params["fit_func_params"],
    ):
        self._fitted_detect_func_params = {}
        if fit_func is not None:
            self._need_fit = True
        else:
            self._need_fit = False
        super().__init__(
            detect_func=detect_func,
            detect_func_params=detect_func_params,
            fit_func=fit_func,
            fit_func_params=fit_func_params,
        )

    def _fit_core(self, s):
        if self.fit_func is not None:
            if self.fit_func_params is not None:
                fit_func_params = self.fit_func_params
            else:
                fit_func_params = {}
            self._fitted_detect_func_params = self.fit_func(
                s, **fit_func_params
            )

    def _predict_core(self, s):
        if self.detect_func_params is not None:
            detect_func_params = self.detect_func_params
        else:
            detect_func_params = {}
        if self.fit_func is not None:
            return self.detect_func(
                s, **{**self._fitted_detect_func_params, **detect_func_params}
            )
        else:
            return self.detect_func(s, **detect_func_params)

    @property
    def fit_func(self):
        return self._fit_func

    @fit_func.setter
    def fit_func(self, value):
        self._fit_func = value
        if value is None:
            self._need_fit = False
        else:
            self._need_fit = True


class ThresholdAD(_Detector1D):
    """Detector that detects anomaly based on user-given threshold.

    This detector compares time series values with user-given thresholds, and
    identifies time points as anomalous when values are beyond the thresholds.

    Parameters
    ----------
    low: float, optional
        Threshold below which a value is regarded anomaly. Default: None, i.e.
        no threshold on lower side.

    high: float, optional
        Threshold above which a value is regarded anomaly. Default: None, i.e.
        no threshold on lower side.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _need_fit = False
    _default_params = {"low": None, "high": None}

    def __init__(
        self, low=_default_params["low"], high=_default_params["high"]
    ):
        super().__init__(low=low, high=high)

    def _fit_core(self, s):
        pass

    def _predict_core(self, s):
        predicted = (
            s > (self.high if (self.high is not None) else float("inf"))
        ) | (s < (self.low if (self.low is not None) else -float("inf")))
        predicted[s.isna()] = np.nan
        return predicted


class QuantileAD(_Detector1D):
    """Detector that detects anomaly based on quantiles of historical data.

    This detector compares time series values with user-specified quantiles
    of historical data, and identifies time points as anomalous when values
    are beyond the thresholds.

    Parameters
    ----------
    low: float, optional
        Quantile of historical data lower which a value is regarded as anomaly.
        Must between 0 and 1. Default: None, i.e. no threshold on lower side.

    high: float, optional
        Quantile of historical data above which a value is regarded as anomaly.
        Must between 0 and 1. Default: None, i.e. no threshold on upper side.

    Attributes
    ----------
    abs_low_: float
        The fitted lower bound of normal range.

    abs_high_: float
        The fitted upper bound of normal range.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _default_params = {"low": None, "high": None}

    def __init__(
        self, low=_default_params["low"], high=_default_params["low"]
    ):
        super().__init__(low=low, high=high)

    def _fit_core(self, s):
        if s.count() == 0:
            raise RuntimeError("Valid values are not enough for training.")
        if self.high is None:
            self.abs_high_ = float("inf")
        else:
            self.abs_high_ = s.quantile(self.high)
        if self.low is None:
            self.abs_low_ = -float("inf")
        else:
            self.abs_low_ = s.quantile(self.low)

    def _predict_core(self, s):
        predicted = (s > self.abs_high_) | (s < self.abs_low_)
        predicted[s.isna()] = np.nan
        return predicted


class InterQuartileRangeAD(_Detector1D):
    """
    Detector that detects anomaly based on inter-quartile range of historical
    data.

    This detector compares time series values with 1st and 3rd quartiles of
    historical data, and identifies time points as anomalous when differences
    are beyond the inter-quartile range times a user-given factor c.

    Parameters
    ----------
    c: float, or 2-tuple (float, float), optional
        Factor used to determine the bound of normal range (betweeen Q1-c*IQR
        and Q3+c*IQR). If a tuple (c1, c2), the factors are for lower and upper
        bound respectively. Default: 3.0.

    Attributes
    ----------
    abs_low_: float
        The fitted lower bound of normal range.

    abs_high_: float
        The fitted upper bound of normal range.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _default_params = {"c": 3.0}

    def __init__(self, c=_default_params["c"]):
        super().__init__(c=c)

    def _fit_core(self, s):
        if s.count() == 0:
            raise RuntimeError("Valid values are not enough for training.")
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        self.abs_low_ = (
            (
                q1
                - iqr
                * (self.c if (not isinstance(self.c, tuple)) else self.c[0])
            )
            if (
                (self.c if (not isinstance(self.c, tuple)) else self.c[0])
                is not None
            )
            else -float("inf")
        )
        self.abs_high_ = (
            q3
            + iqr * (self.c if (not isinstance(self.c, tuple)) else self.c[1])
            if (
                (self.c if (not isinstance(self.c, tuple)) else self.c[1])
                is not None
            )
            else float("inf")
        )

    def _predict_core(self, s):
        predicted = (s > self.abs_high_) | (s < self.abs_low_)
        predicted[s.isna()] = np.nan
        return predicted


class GeneralizedESDTestAD(_Detector1D):
    """Detector that detects anomaly based on generalized ESD test.

    This detector performs generalized extreme Studentized deviate (ESD) test
    [1, 2] on historical data and identifies normal values vs. outliers for
    training. For predicting, the detector adds each value in the testing
    series to the set of normal values from training series independently, and
    performs generalized ESD test to this set (all normal values from training
    series, plus one value from testing series) to evaluate if this value of
    interest is an outlier.

    Please note a key assumption of generalized ESD test is that normal values
    follow an approximately normal distribution. Please only use this detector
    when this assumption holds.

    [1] Rosner, Bernard (May 1983), Percentage Points for a Generalized ESD
    Many-Outlier Procedure,Technometrics, 25(2), pp. 165-172.

    [2] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

    Parameters
    ----------
    alpha: float, optional
        Significance level. Default: 0.05.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.
    """

    _default_params = {"alpha": 0.05}

    def __init__(self, alpha=_default_params["alpha"]):
        super().__init__(alpha=alpha)

    def _fit_core(self, s):
        if s.count() == 0:
            raise RuntimeError("Valid values are not enough for training.")
        R = pd.Series(np.zeros(len(s)), index=s.index)
        n = s.count()
        Lambda = pd.Series(np.zeros(len(s)), index=s.index)
        s_copy = s.copy()
        i = 0
        while s_copy.count() > 0:
            i += 1
            ind = (s_copy - s_copy.mean()).abs().idxmax()
            R[ind] = (
                abs(s_copy[ind] - s_copy.mean()) / s_copy.std()
                if s_copy.std() > 0
                else 0
            )
            s_copy[ind] = np.nan
            p = 1 - self.alpha / (2 * (n - i + 1))
            Lambda[ind] = (
                (n - i)
                * t.ppf(p, n - i - 1)
                / np.sqrt((n - i - 1 + t.ppf(p, n - i - 1) ** 2) * (n - i + 1))
            )
            if R[ind] <= Lambda[ind]:
                break
        self._normal_sum = s[Lambda >= R].sum()
        self._normal_squared_sum = (s[Lambda >= R] ** 2).sum()
        self._normal_count = s[Lambda >= R].count()
        i = 1
        n = self._normal_count + 1
        p = 1 - self.alpha / (2 * (n - i + 1))
        self._lambda = (
            (n - i)
            * t.ppf(p, n - i - 1)
            / np.sqrt((n - i - 1 + t.ppf(p, n - i - 1) ** 2) * (n - i + 1))
        )

    def _predict_core(self, s):
        new_sum = s + self._normal_sum
        new_count = self._normal_count + 1
        new_mean = new_sum / new_count
        new_squared_sum = s ** 2 + self._normal_squared_sum
        new_std = np.sqrt(
            (
                new_squared_sum
                - 2 * new_mean * new_sum
                + new_count * new_mean ** 2
            )
            / (new_count - 1)
        )
        predicted = (s - new_mean).abs() / new_std > self._lambda
        predicted[s.isna()] = np.nan
        return predicted


# =============================================================================
# PLEASE PUT PIPE-DERIVED DETECTOR CLASSES BELOW THIS LINE
# =============================================================================


class PersistAD(_Detector1D):
    """Detector that detects anomaly based on values in a preceding period.

    This detector compares time series values with the values of their
    preceding time windows, and identifies a time point as anomalous if the
    change of value from its preceding average or median is beyond a threshold
    based on historical interquartile range.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    window: int, optional
        Number of time points in the time window. Default: 1.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 3.0.

    side: str, optional
        If "both", to detect anomalous positive and negative changes;
        If "positive", to only detect anomalous positive changes;
        If "negative", to only detect anomalous negative changes.
        Default: "both".

    min_periods: int, optional
        Minimum number of observations in each window required to have a value
        for that window. Default: None, i.e. all observations must have values.

    agg: str, optional
        Aggregation operation of the time window, either "mean" or "median".
        Default: "median".

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.

    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _default_params = {
        "window": 1,
        "c": 3.0,
        "side": "both",
        "min_periods": None,
        "agg": "median",
    }

    def __init__(
        self,
        window=_default_params["window"],
        c=_default_params["c"],
        side=_default_params["side"],
        min_periods=_default_params["min_periods"],
        agg=_default_params["agg"],
    ):
        self.pipe_ = Pipenet(
            [
                {
                    "name": "diff_abs",
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=(window, 1),
                        center=True,
                        min_periods=(min_periods, 1),
                        diff="l1",
                    ),
                    "input": "original",
                },
                {
                    "name": "iqr_ad",
                    "model": InterQuartileRangeAD(c=(None, c)),
                    "input": "diff_abs",
                },
                {
                    "name": "diff",
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=(window, 1),
                        center=True,
                        min_periods=(min_periods, 1),
                        diff="diff",
                    ),
                    "input": "original",
                },
                {
                    "name": "sign_check",
                    "model": ThresholdAD(
                        high=(
                            0.0
                            if side == "positive"
                            else (
                                float("inf")
                                if side == "negative"
                                else -float("inf")
                            )
                        ),
                        low=(
                            0.0
                            if side == "negative"
                            else (
                                -float("inf")
                                if side == "positive"
                                else float("inf")
                            )
                        ),
                    ),
                    "input": "diff",
                },
                {
                    "name": "and",
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            ]
        )
        super().__init__(
            c=c, side=side, window=window, min_periods=min_periods, agg=agg
        )
        self._sync_params()

    def _sync_params(self):
        if self.agg not in ["median", "mean"]:
            raise ValueError(
                "Parameter `agg` must be either 'median' or 'mean'."
            )
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps[0]["model"].agg = self.agg
        self.pipe_.steps[0]["model"].window = (self.window, 1)
        self.pipe_.steps[0]["model"].min_periods = (self.min_periods, 1)
        self.pipe_.steps[1]["model"].c = (None, self.c)
        self.pipe_.steps[2]["model"].agg = self.agg
        self.pipe_.steps[2]["model"].window = (self.window, 1)
        self.pipe_.steps[2]["model"].min_periods = (self.min_periods, 1)
        self.pipe_.steps[3]["model"].high = (
            0.0
            if self.side == "positive"
            else (float("inf") if self.side == "negative" else -float("inf"))
        )
        self.pipe_.steps[3]["model"].low = (
            0.0
            if self.side == "negative"
            else (-float("inf") if self.side == "positive" else float("inf"))
        )

    def _fit_core(self, s):
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s):
        self._sync_params()
        return self.pipe_.detect(s)


class LevelShiftAD(_Detector1D):
    """Detector that detects level shift of time series values.

    This detector compares median values inside time windows next to each
    others, and identifies a time point as a level shift point if difference
    between time windows on its left-side and its right-side is beyond a
    threshold based on historical interquartile range.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    window: int, optional
        Number of time points in each time window. Default: 10.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 6.0.

    side: str, optional
        If "both", to detect anomalous positive and negative changes;
        If "positive", to only detect anomalous positive changes;
        If "negative", to only detect anomalous negative changes.
        Default: "both".

    min_periods: int, optional
        Minimum number of observations in each window required to have a value
        for that window. Default: None, i.e. all observations must have values.

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.

    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _default_params = {
        "window": 10,
        "c": 6.0,
        "side": "both",
        "min_periods": None,
    }

    def __init__(
        self,
        window=_default_params["window"],
        c=_default_params["c"],
        side=_default_params["side"],
        min_periods=_default_params["min_periods"],
    ):
        self.pipe_ = Pipenet(
            [
                {
                    "name": "diff_abs",
                    "model": DoubleRollingAggregate(
                        agg="median",
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="l1",
                    ),
                    "input": "original",
                },
                {
                    "name": "iqr_ad",
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "diff_abs",
                },
                {
                    "name": "diff",
                    "model": DoubleRollingAggregate(
                        agg="median",
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="diff",
                    ),
                    "input": "original",
                },
                {
                    "name": "sign_check",
                    "model": ThresholdAD(
                        high=(
                            0.0
                            if side == "positive"
                            else (
                                float("inf")
                                if side == "negative"
                                else -float("inf")
                            )
                        ),
                        low=(
                            0.0
                            if side == "negative"
                            else (
                                -float("inf")
                                if side == "positive"
                                else float("inf")
                            )
                        ),
                    ),
                    "input": "diff",
                },
                {
                    "name": "and",
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            ]
        )
        super().__init__(
            c=c, side=side, window=window, min_periods=min_periods
        )
        self._sync_params()

    def _sync_params(self):
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps[0]["model"].window = self.window
        self.pipe_.steps[0]["model"].min_periods = self.min_periods
        self.pipe_.steps[1]["model"].c = (None, self.c)
        self.pipe_.steps[2]["model"].window = self.window
        self.pipe_.steps[2]["model"].min_periods = self.min_periods
        self.pipe_.steps[3]["model"].high = (
            0.0
            if self.side == "positive"
            else (float("inf") if self.side == "negative" else -float("inf"))
        )
        self.pipe_.steps[3]["model"].low = (
            0.0
            if self.side == "negative"
            else (-float("inf") if self.side == "positive" else float("inf"))
        )

    def _fit_core(self, s):
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s):
        self._sync_params()
        return self.pipe_.detect(s)


class VolatilityShiftAD(_Detector1D):
    """Detector that detects level shift of time series volatility.

    This detector compares standard deviations inside time windows next to each
    others, and identifies a time point as a volatility shift point if change
    over time windows from its left-side to its right-side is beyond a
    threshold based on historical interquartile range.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    window: int, optional
        Number of time points in each time window. Default: 10.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 6.0.

    side: str, optional
        If "both", to detect anomalous positive and negative changes;
        If "positive", to only detect anomalous positive changes;
        If "negative", to only detect anomalous negative changes.
        Default: "both".

    min_periods: int, optional
        Minimum number of observations in each window required to have a value
        for that window. Default: None, i.e. all observations must have values.

    agg: str, optional
        Aggregation operation of the time window, one of "std", "iqr" or "idr".
        Default: "std".

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _default_params = {
        "window": 10,
        "c": 6.0,
        "side": "both",
        "min_periods": None,
        "agg": "std",
    }

    def __init__(
        self,
        window=_default_params["window"],
        c=_default_params["c"],
        side=_default_params["side"],
        min_periods=_default_params["min_periods"],
        agg=_default_params["agg"],
    ):
        self.pipe_ = Pipenet(
            [
                {
                    "name": "diff_abs",
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="abs_rel_diff",
                    ),
                    "input": "original",
                },
                {
                    "name": "iqr_ad",
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "diff_abs",
                },
                {
                    "name": "diff",
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="diff",
                    ),
                    "input": "original",
                },
                {
                    "name": "sign_check",
                    "model": ThresholdAD(
                        high=(
                            0.0
                            if side == "positive"
                            else (
                                float("inf")
                                if side == "negative"
                                else -float("inf")
                            )
                        ),
                        low=(
                            0.0
                            if side == "negative"
                            else (
                                -float("inf")
                                if side == "positive"
                                else float("inf")
                            )
                        ),
                    ),
                    "input": "diff",
                },
                {
                    "name": "and",
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            ]
        )
        super().__init__(
            agg=agg, c=c, side=side, window=window, min_periods=min_periods
        )
        self._sync_params()

    def _sync_params(self):
        if self.agg not in ["std", "iqr", "idr"]:
            raise ValueError("Parameter `agg` must be 'std', 'iqr' or 'idr'.")
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps[0]["model"].agg = self.agg
        self.pipe_.steps[0]["model"].window = self.window
        self.pipe_.steps[0]["model"].min_periods = self.min_periods
        self.pipe_.steps[1]["model"].c = (None, self.c)
        self.pipe_.steps[2]["model"].agg = self.agg
        self.pipe_.steps[2]["model"].window = self.window
        self.pipe_.steps[2]["model"].min_periods = self.min_periods
        self.pipe_.steps[3]["model"].high = (
            0.0
            if self.side == "positive"
            else (float("inf") if self.side == "negative" else -float("inf"))
        )
        self.pipe_.steps[3]["model"].low = (
            0.0
            if self.side == "negative"
            else (-float("inf") if self.side == "positive" else float("inf"))
        )

    def _fit_core(self, s):
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s):
        self._sync_params()
        return self.pipe_.detect(s)


class AutoregressionAD(_Detector1D):
    """Detector that detects anomalous autoregression property in time series.

    Many time series has autoregression behavior. For example, in a linear
    autoregression time series, current value is a linear combination of
    serveral previous values. Violation of usual autoregression behavior may
    indicate anomaly.

    The detector applies a regressor to learn autoregression property of the
    time series, and identifies a time point as anomalous when the residual of
    autoregression is beyond a threshold based on historical interquartile
    range.

    This detector is internally implemented aattribute `pipe_`.nced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    n_steps: int, optional
        Number of steps (previous values) to include in the model. Default: 1.

    step_size: int, optional
        Length of a step. For example, if n_steps=2, step_size=3, X_[t-3] and
        X_[t-6] will be used to predict X_[t]. Default: 1.

    regressor: object, optional
        Regressor to be used. Same as a scikit-learn regressor, it should
        minimally have `fit` and `predict` methods. If not given, a linear
        regressor will be used.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 3.0.

    side: str, optional
        If "both", to detect anomalous positive and negative residuals;
        If "positive", to only detect anomalous positive residuals;
        If "negative", to only detect anomalous negative residuals.
        Default: "both".

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _default_params = {
        "n_steps": 1,
        "step_size": 1,
        "regressor": None,
        "c": 3.0,
        "side": "both",
    }

    def __init__(
        self,
        n_steps=_default_params["n_steps"],
        step_size=_default_params["step_size"],
        regressor=_default_params["regressor"],
        c=_default_params["c"],
        side=_default_params["side"],
    ):
        if regressor is None:
            regressor = LinearRegression()
        self.pipe_ = Pipenet(
            [
                {
                    "name": "retrospetive",
                    "model": Retrospect(
                        n_steps=n_steps + 1, step_size=step_size
                    ),
                    "input": "original",
                },
                {
                    "name": "regression_residual",
                    "model": RegressionResidual(regressor=regressor),
                    "input": "retrospetive",
                },
                {
                    "name": "abs_residual",
                    "model": CustomizedTransformer1D(transform_func=abs),
                    "input": "regression_residual",
                },
                {
                    "name": "iqr_ad",
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "abs_residual",
                },
                {
                    "name": "sign_check",
                    "model": ThresholdAD(
                        high=(
                            0.0
                            if side == "positive"
                            else (
                                float("inf")
                                if side == "negative"
                                else -float("inf")
                            )
                        ),
                        low=(
                            0.0
                            if side == "negative"
                            else (
                                -float("inf")
                                if side == "positive"
                                else float("inf")
                            )
                        ),
                    ),
                    "input": "regression_residual",
                },
                {
                    "name": "and",
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            ]
        )
        super().__init__(
            n_steps=n_steps,
            step_size=step_size,
            regressor=regressor,
            c=c,
            side=side,
        )
        self._sync_params()

    def _sync_params(self):
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps[0]["model"].n_steps = self.n_steps + 1
        self.pipe_.steps[0]["model"].step_size = self.step_size
        self.pipe_.steps[1]["model"].regressor = self.regressor
        self.pipe_.steps[3]["model"].c = (None, self.c)
        self.pipe_.steps[4]["model"].high = (
            0.0
            if self.side == "positive"
            else (float("inf") if self.side == "negative" else -float("inf"))
        )
        self.pipe_.steps[4]["model"].low = (
            0.0
            if self.side == "negative"
            else (-float("inf") if self.side == "positive" else float("inf"))
        )

    def _fit_core(self, s):
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s):
        self._sync_params()
        return self.pipe_.detect(s)


class SeasonalAD(_Detector1D):
    """Detector that detects anomalous values away from seasonal pattern.

    This detector uses a seasonal decomposition transformer to remove seasonal
    pattern (as well as trend if STL method is selected), and identifiess a
    time point as anomalous when the residual of seasonal decomposition is
    beyond a threshold based on historical interquartile range.

    This detector is internally implemented aattribute `pipe_`.nced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    method: str, optional
        If 'naive',  use naive seasonal decomposition;
        If 'stl', use STL method.
        See `adtk.transformer_1d.NaiveSeasonalDecomposition` and
        `adtk.transformer_1d.STLDecomposition` for more details.
        Default: 'naive'.

    freq: int, optional
        Length of a seasonal cycle. If not given, the model will determine
        automatically based on autocorrelation of the training series. Default:
        None.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 3.0.

    side: str, optional
        If "both", to detect anomalous positive and negative residuals;
        If "positive", to only detect anomalous positive residuals;
        If "negative", to only detect anomalous negative residuals.
        Default: "both".

    Attributes
    ----------
    freq_: int
        Length of seasonal cycle. Equal to parameter `freq` if it is given.
        Otherwise, calculated based on autocorrelation of the training series.

    seasonal_: pandas.Series
        Seasonal pattern extracted from training series.

    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.


    This is an univariate detector. When it is applied to a multivariate time
    series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _default_params = {
        "method": "naive",
        "freq": None,
        "side": "both",
        "c": 3.0,
    }

    def __init__(
        self,
        method=_default_params["method"],
        freq=_default_params["freq"],
        side=_default_params["side"],
        c=_default_params["c"],
    ):
        self.pipe_ = Pipenet(
            [
                {
                    "name": "deseasonal_residual",
                    "model": (
                        NaiveSeasonalDecomposition(freq=freq)
                        if method == "naive"
                        else STLDecomposition(freq=freq)
                    ),
                    "input": "original",
                },
                {
                    "name": "abs_residual",
                    "model": CustomizedTransformer1D(transform_func=abs),
                    "input": "deseasonal_residual",
                },
                {
                    "name": "iqr_ad",
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "abs_residual",
                },
                {
                    "name": "sign_check",
                    "model": ThresholdAD(
                        high=(
                            0.0
                            if side == "positive"
                            else (
                                float("inf")
                                if side == "negative"
                                else -float("inf")
                            )
                        ),
                        low=(
                            0.0
                            if side == "negative"
                            else (
                                -float("inf")
                                if side == "positive"
                                else float("inf")
                            )
                        ),
                    ),
                    "input": "deseasonal_residual",
                },
                {
                    "name": "and",
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            ]
        )
        super().__init__(method=method, freq=freq, side=side, c=c)
        self._sync_params()

    def _sync_params(self):
        if self.method not in ["naive", "stl"]:
            raise ValueError("Parameter `method` must be 'naive' or 'stl'.")
        if (self.method == "naive") and (
            self.pipe_.steps[0]["model"].__class__
            != NaiveSeasonalDecomposition
        ):
            self.pipe_.steps[0]["model"] = NaiveSeasonalDecomposition()
        if (self.method == "stl") and (
            self.pipe_.steps[0]["model"].__class__ != STLDecomposition
        ):
            self.pipe_.steps[0]["model"] = STLDecomposition()
        self.pipe_.steps[0]["model"].freq = self.freq
        self.pipe_.steps[2]["model"].c = (None, self.c)
        self.pipe_.steps[3]["model"].high = (
            0.0
            if self.side == "positive"
            else (float("inf") if self.side == "negative" else -float("inf"))
        )
        self.pipe_.steps[3]["model"].low = (
            0.0
            if self.side == "negative"
            else (-float("inf") if self.side == "positive" else float("inf"))
        )

    def _fit_core(self, s):
        self._sync_params()
        self.pipe_.fit(s)
        self.freq_ = self.pipe_.steps[0]["model"].freq_
        self.seasonal_ = self.pipe_.steps[0]["model"].seasonal_

    def _predict_core(self, s):
        self._sync_params()
        return self.pipe_.detect(s)
