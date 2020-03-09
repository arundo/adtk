"""Module for 1-dimensional detectors.

1-dimensional detectors detect anomalies from 1-dimensional time series, i.e.
from pandas Series.

"""

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy.stats import t
from sklearn.linear_model import LinearRegression

from .._detector_base import (
    _NonTrainableUnivariateDetector,
    _TrainableUnivariateDetector,
)
from ..aggregator import AndAggregator
from ..pipe import Pipenet
from ..transformer import (
    ClassicSeasonalDecomposition,
    CustomizedTransformer1D,
    DoubleRollingAggregate,
    RegressionResidual,
    Retrospect,
)


class CustomizedDetector1D(_TrainableUnivariateDetector):
    """Univariate detector derived from a user-given function and parameters.

    Parameters
    ----------
    detect_func: function
        A function detecting anomalies from univariate time series.

        The first input argument must be a pandas Series, optional input
        argument may be accepted through parameter `detect_func_params` and the
        output of `fit_func`, and the output must be a binary pandas Series
        with the same index as input.

    detect_func_params: dict, optional
        Parameters of `detect_func`. Default: None.

    fit_func: function, optional
        A function training parameters of `detect_func` with univariate time
        series.

        The first input argument must be a pandas Series, optional input
        argument may be accepted through parameter `fit_func_params`, and the
        output must be a dict that can be used by `detect_func` as parameters.
        Default: None.

    fit_func_params: dict, optional
        Parameters of `fit_func`. Default: None.

    """

    def __init__(
        self,
        detect_func: Callable,
        detect_func_params: Optional[Dict[str, Any]] = None,
        fit_func: Optional[Callable] = None,
        fit_func_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._fitted_detect_func_params = {}  # type: Dict[str, Any]
        super().__init__()
        self.detect_func = detect_func
        self.detect_func_params = detect_func_params
        self.fit_func = fit_func
        self.fit_func_params = fit_func_params
        if self.fit_func is None:
            self._fitted = 1

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return (
            "detect_func",
            "detect_func_params",
            "fit_func",
            "fit_func_params",
        )

    def _fit_core(self, s: pd.Series) -> None:
        if self.fit_func is not None:
            if self.fit_func_params is not None:
                fit_func_params = self.fit_func_params
            else:
                fit_func_params = {}
            self._fitted_detect_func_params = self.fit_func(
                s, **fit_func_params
            )

    def _predict_core(self, s: pd.Series) -> pd.Series:
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


class ThresholdAD(_NonTrainableUnivariateDetector):
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
        no threshold on upper side.

    """

    def __init__(
        self, low: Optional[float] = None, high: Optional[float] = None
    ) -> None:
        super().__init__()
        self.low = low
        self.high = high

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("low", "high")

    def _predict_core(self, s: pd.Series) -> pd.Series:
        predicted = (
            s > (self.high if (self.high is not None) else float("inf"))
        ) | (s < (self.low if (self.low is not None) else -float("inf")))
        predicted[s.isna()] = np.nan
        return predicted


class QuantileAD(_TrainableUnivariateDetector):
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

    """

    def __init__(
        self, low: Optional[float] = None, high: Optional[float] = None
    ) -> None:
        super().__init__()
        self.low = low
        self.high = high

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("low", "high")

    def _fit_core(self, s: pd.Series) -> None:
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

    def _predict_core(self, s: pd.Series) -> pd.Series:
        predicted = (s > self.abs_high_) | (s < self.abs_low_)
        predicted[s.isna()] = np.nan
        return predicted


class InterQuartileRangeAD(_TrainableUnivariateDetector):
    """
    Detector that detects anomaly based on inter-quartile range of historical
    data.

    This detector compares time series values with 1st and 3rd quartiles of
    historical data, and identifies time points as anomalous when differences
    are beyond the inter-quartile range (IQR) times a user-given factor c.

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

    """

    def __init__(
        self,
        c: Union[
            Optional[float], Tuple[Optional[float], Optional[float]]
        ] = 3.0,
    ) -> None:
        super().__init__()
        self.c = c

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("c",)

    def _fit_core(self, s: pd.Series) -> None:
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

    def _predict_core(self, s: pd.Series) -> pd.Series:
        predicted = (s > self.abs_high_) | (s < self.abs_low_)
        predicted[s.isna()] = np.nan
        return predicted


class GeneralizedESDTestAD(_TrainableUnivariateDetector):
    """Detector that detects anomaly based on generalized ESD test.

    This detector performs generalized extreme Studentized deviate (ESD) test
    [1, 2] on historical data and identifies normal values vs. outliers for
    training. For predicting, the detector adds each value in the testing
    series to the set of normal values from training series independently, and
    performs generalized ESD test to this set (all normal values from training
    series, plus one value from testing series) to evaluate if this value of
    interest is an outlier.

    Please note a key assumption of generalized ESD test is that values follow
    an approximately normal distribution. Please only use this detector when
    this assumption holds.

    [1] Rosner, Bernard (May 1983), Percentage Points for a Generalized ESD
    Many-Outlier Procedure,Technometrics, 25(2), pp. 165-172.

    [2] https://www.itl.nist.gov/div898/handbook/eda/section3/eda35h3.htm

    Parameters
    ----------
    alpha: float, optional
        Significance level. Default: 0.05.

    """

    def __init__(self, alpha: float = 0.05) -> None:
        super().__init__()
        self.alpha = alpha

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("alpha",)

    def _fit_core(self, s: pd.Series) -> None:
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

    def _predict_core(self, s: pd.Series) -> pd.Series:
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


class PersistAD(_TrainableUnivariateDetector):
    """Detector that detects anomaly based on values in a preceding period.

    This detector compares time series values with the values of their
    preceding time windows, and identifies a time point as anomalous if the
    change of value from its preceding average or median is anomalously large.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    window: int or str, optional
        Size of the preceding time window.

        - If int, it is the number of time point in this time window.
        - If str, it must be able to be converted into a pandas Timedelta
          object.

        Default: 1.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 3.0.

    side: str, optional
        - If "both", to detect anomalous positive and negative changes;
        - If "positive", to only detect anomalous positive changes;
        - If "negative", to only detect anomalous negative changes.

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

    """

    def __init__(
        self,
        window: Union[int, str] = 1,
        c: float = 3.0,
        side: str = "both",
        min_periods: Optional[int] = None,
        agg: str = "median",
    ) -> None:
        self.pipe_ = Pipenet(
            {
                "diff_abs": {
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=(window, 1),
                        center=True,
                        min_periods=(min_periods, 1),
                        diff="l1",
                    ),
                    "input": "original",
                },
                "iqr_ad": {
                    "model": InterQuartileRangeAD(c=(None, c)),
                    "input": "diff_abs",
                },
                "diff": {
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=(window, 1),
                        center=True,
                        min_periods=(min_periods, 1),
                        diff="diff",
                    ),
                    "input": "original",
                },
                "sign_check": {
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
                "and": {
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            }
        )
        super().__init__()
        self.c = c
        self.side = side
        self.window = window
        self.min_periods = min_periods
        self.agg = agg
        self._sync_params()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("window", "c", "side", "min_periods", "agg")

    def _sync_params(self) -> None:
        if self.agg not in ["median", "mean"]:
            raise ValueError(
                "Parameter `agg` must be either 'median' or 'mean'."
            )
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps["diff_abs"]["model"].set_params(
            agg=self.agg,
            window=(self.window, 1),
            min_periods=(self.min_periods, 1),
        )
        self.pipe_.steps["iqr_ad"]["model"].set_params(c=(None, self.c))
        self.pipe_.steps["diff"]["model"].set_params(
            agg=self.agg,
            window=(self.window, 1),
            min_periods=(self.min_periods, 1),
        )
        self.pipe_.steps["sign_check"]["model"].set_params(
            high=(
                0.0
                if self.side == "positive"
                else (
                    float("inf") if self.side == "negative" else -float("inf")
                )
            ),
            low=(
                0.0
                if self.side == "negative"
                else (
                    -float("inf") if self.side == "positive" else float("inf")
                )
            ),
        )

    def _fit_core(self, s: pd.Series) -> None:
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s: pd.Series) -> pd.Series:
        self._sync_params()
        return self.pipe_.detect(s)


class LevelShiftAD(_TrainableUnivariateDetector):
    """Detector that detects level shift of time series values.

    This detector compares values of two time windows next to each others, and
    identifies the time point in between as an level-shift point if the
    difference of the medians in the two time windows is anomalously large.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    window: int or str, or 2-tuple of int or str
        Size of the time windows.

        - If int, it is the number of time point in this time window.
        - If str, it must be able to be converted into a pandas Timedelta
          object.
        - If 2-tuple, it defines the left and right window respectively.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 6.0.

    side: str, optional
        - If "both", to detect anomalous positive and negative changes;
        - If "positive", to only detect anomalous positive changes;
        - If "negative", to only detect anomalous negative changes.

        Default: "both".

    min_periods: int, or 2-tuple of int, optional
        Minimum number of observations in each window required to have a value
        for that window. If 2-tuple, it defines the left and right window
        respectively. Default: None, i.e. all observations must have values.

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.

    """

    def __init__(
        self,
        window: Union[
            Union[int, str], Tuple[Union[int, str], Union[int, str]]
        ],
        c: float = 6.0,
        side: str = "both",
        min_periods: Union[
            Optional[int], Tuple[Optional[int], Optional[int]]
        ] = None,
    ) -> None:
        self.pipe_ = Pipenet(
            {
                "diff_abs": {
                    "model": DoubleRollingAggregate(
                        agg="median",
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="l1",
                    ),
                    "input": "original",
                },
                "iqr_ad": {
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "diff_abs",
                },
                "diff": {
                    "model": DoubleRollingAggregate(
                        agg="median",
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="diff",
                    ),
                    "input": "original",
                },
                "sign_check": {
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
                "and": {
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            }
        )
        super().__init__()
        self.c = c
        self.side = side
        self.window = window
        self.min_periods = min_periods
        self._sync_params()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("window", "c", "side", "min_periods")

    def _sync_params(self) -> None:
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps["diff_abs"]["model"].set_params(
            window=self.window, min_periods=self.min_periods
        )
        self.pipe_.steps["iqr_ad"]["model"].set_params(c=(None, self.c))
        self.pipe_.steps["diff"]["model"].set_params(
            window=self.window, min_periods=self.min_periods
        )
        self.pipe_.steps["sign_check"]["model"].set_params(
            high=(
                0.0
                if self.side == "positive"
                else (
                    float("inf") if self.side == "negative" else -float("inf")
                )
            ),
            low=(
                0.0
                if self.side == "negative"
                else (
                    -float("inf") if self.side == "positive" else float("inf")
                )
            ),
        )

    def _fit_core(self, s: pd.Series) -> None:
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s: pd.Series) -> pd.Series:
        self._sync_params()
        return self.pipe_.detect(s)


class VolatilityShiftAD(_TrainableUnivariateDetector):
    """Detector that detects shift of volatility in time series.

    This detector compares volatility of two time windows next to each others,
    and identifies the time point in between as a volatility-shift point if the
    difference of the volatility measurement in the two time windows is
    anomalously large.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    window: int or str, or 2-tuple of int or str
        Size of the time windows.

        - If int, it is the number of time point in this time window.
        - If str, it must be able to be converted into a pandas Timedelta
          object.
        - If 2-tuple, it defines the left and right window respectively.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 6.0.

    side: str, optional
        - If "both", to detect anomalous positive and negative changes;
        - If "positive", to only detect anomalous positive changes;
        - If "negative", to only detect anomalous negative changes.

        Default: "both".

    min_periods: int, optional
        Minimum number of observations in each window required to have a value
        for that window. If 2-tuple, it defines the left and right window
        respectively. Default: None, i.e. all observations must have values.

    agg: str, optional
        Measurement of volatility in a time window, one of "std" (standard
        deviation), "iqr" (interquartile range), or "idr" (interdecile range).
        Default: "std".

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.

    """

    def __init__(
        self,
        window: Union[
            Union[int, str], Tuple[Union[int, str], Union[int, str]]
        ],
        c: float = 6.0,
        side: str = "both",
        min_periods: Union[
            Optional[int], Tuple[Optional[int], Optional[int]]
        ] = None,
        agg: str = "std",
    ) -> None:
        self.pipe_ = Pipenet(
            {
                "diff_abs": {
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="abs_rel_diff",
                    ),
                    "input": "original",
                },
                "iqr_ad": {
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "diff_abs",
                },
                "diff": {
                    "model": DoubleRollingAggregate(
                        agg=agg,
                        window=window,
                        center=True,
                        min_periods=min_periods,
                        diff="diff",
                    ),
                    "input": "original",
                },
                "sign_check": {
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
                "and": {
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            }
        )
        super().__init__()
        self.agg = agg
        self.c = c
        self.side = side
        self.window = window
        self.min_periods = min_periods
        self._sync_params()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("window", "c", "side", "min_periods", "agg")

    def _sync_params(self) -> None:
        if self.agg not in ["std", "iqr", "idr"]:
            raise ValueError("Parameter `agg` must be 'std', 'iqr' or 'idr'.")
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps["diff_abs"]["model"].set_params(
            agg=self.agg, window=self.window, min_periods=self.min_periods
        )
        self.pipe_.steps["iqr_ad"]["model"].set_params(c=(None, self.c))
        self.pipe_.steps["diff"]["model"].set_params(
            agg=self.agg, window=self.window, min_periods=self.min_periods
        )
        self.pipe_.steps["sign_check"]["model"].set_params(
            high=(
                0.0
                if self.side == "positive"
                else (
                    float("inf") if self.side == "negative" else -float("inf")
                )
            ),
            low=(
                0.0
                if self.side == "negative"
                else (
                    -float("inf") if self.side == "positive" else float("inf")
                )
            ),
        )

    def _fit_core(self, s: pd.Series) -> None:
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s: pd.Series) -> pd.Series:
        self._sync_params()
        return self.pipe_.detect(s)


class AutoregressionAD(_TrainableUnivariateDetector):
    """Detector that detects anomalous autoregression property in time series.

    Many time series has autoregressive behavior. For example, in a linearly
    autoregressive time series, current value is a linear combination of
    serveral previous values. Violation of usual autoregressive behavior may
    indicate anomaly.

    The detector applies a regressor to learn autoregressive property of the
    time series, and identifies a time point as anomalous when the residual of
    autoregression is anomalously large.

    This detector is internally implemented as a `Pipenet` object. Advanced
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
        - If "both", to detect anomalous positive and negative residuals;
        - If "positive", to only detect anomalous positive residuals;
        - If "negative", to only detect anomalous negative residuals.

        Default: "both".

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.

    """

    def __init__(
        self,
        n_steps: int = 1,
        step_size: int = 1,
        regressor: Optional[Any] = None,
        c: float = 3.0,
        side: str = "both",
    ) -> None:
        if regressor is None:
            regressor = LinearRegression()
        self.pipe_ = Pipenet(
            {
                "retrospetive": {
                    "model": Retrospect(
                        n_steps=n_steps + 1, step_size=step_size
                    ),
                    "input": "original",
                },
                "regression_residual": {
                    "model": RegressionResidual(
                        target="t-0", regressor=regressor
                    ),
                    "input": "retrospetive",
                },
                "abs_residual": {
                    "model": CustomizedTransformer1D(transform_func=abs),
                    "input": "regression_residual",
                },
                "iqr_ad": {
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "abs_residual",
                },
                "sign_check": {
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
                "and": {
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            }
        )
        super().__init__()
        self.n_steps = n_steps
        self.step_size = step_size
        self.regressor = regressor
        self.c = c
        self.side = side
        self._sync_params()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("n_steps", "step_size", "regressor", "c", "side")

    def _sync_params(self) -> None:
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        if self.regressor is None:
            self.regressor = LinearRegression()
        self.pipe_.steps["retrospetive"]["model"].set_params(
            n_steps=self.n_steps + 1, step_size=self.step_size
        )
        self.pipe_.steps["regression_residual"]["model"].set_params(
            regressor=self.regressor
        )
        self.pipe_.steps["iqr_ad"]["model"].set_params(c=(None, self.c))
        self.pipe_.steps["sign_check"]["model"].set_params(
            high=(
                0.0
                if self.side == "positive"
                else (
                    float("inf") if self.side == "negative" else -float("inf")
                )
            ),
            low=(
                0.0
                if self.side == "negative"
                else (
                    -float("inf") if self.side == "positive" else float("inf")
                )
            ),
        )

    def _fit_core(self, s: pd.Series) -> None:
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s: pd.Series) -> pd.Series:
        self._sync_params()
        return self.pipe_.detect(s)


class SeasonalAD(_TrainableUnivariateDetector):
    """Detector that detects anomalous values away from seasonal pattern.

    This detector uses a seasonal decomposition transformer to remove seasonal
    pattern (as well as trend optional), and identifies a time point as
    anomalous when the residual of seasonal decomposition is anomalously large.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    freq: int, optional
        Length of a seasonal cycle as the number of time points in a cycle. If
        not specified, the model will try to determine it based on
        autocorrelation of the training series. Default: None.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 3.0.

    side: str, optional
        - If "both", to detect anomalous positive and negative residuals;
        - If "positive", to only detect anomalous positive residuals;
        - If "negative", to only detect anomalous negative residuals.

        Default: "both".

    trend: bool, optional
        Whether to extract trend during decomposition. Default: False.

    Attributes
    ----------
    freq_: int
        Length of seasonal cycle as the number of time points in a cycle. Equal
        to parameter `freq` if it is specified. Otherwise, calculated based on
        autocorrelation of the training series.

    seasonal_: pandas.Series
        Seasonal pattern extracted from training series.

    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.

    """

    def __init__(
        self,
        freq: Optional[int] = None,
        side: str = "both",
        c: float = 3.0,
        trend: bool = False,
    ) -> None:
        self.pipe_ = Pipenet(
            {
                "deseasonal_residual": {
                    "model": (
                        ClassicSeasonalDecomposition(freq=freq, trend=trend)
                    ),
                    "input": "original",
                },
                "abs_residual": {
                    "model": CustomizedTransformer1D(transform_func=abs),
                    "input": "deseasonal_residual",
                },
                "iqr_ad": {
                    "model": InterQuartileRangeAD((None, c)),
                    "input": "abs_residual",
                },
                "sign_check": {
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
                "and": {
                    "model": AndAggregator(),
                    "input": ["iqr_ad", "sign_check"],
                },
            }
        )
        super().__init__()
        self.freq = freq
        self.side = side
        self.c = c
        self.trend = trend
        self._sync_params()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("freq", "side", "c", "trend")

    def _sync_params(self) -> None:
        self.pipe_.steps["deseasonal_residual"]["model"].set_params(
            freq=self.freq, trend=self.trend
        )
        self.pipe_.steps["iqr_ad"]["model"].set_params(c=(None, self.c))
        self.pipe_.steps["sign_check"]["model"].set_params(
            high=(
                0.0
                if self.side == "positive"
                else (
                    float("inf") if self.side == "negative" else -float("inf")
                )
            ),
            low=(
                0.0
                if self.side == "negative"
                else (
                    -float("inf") if self.side == "positive" else float("inf")
                )
            ),
        )

    def _fit_core(self, s: pd.Series) -> None:
        self._sync_params()
        self.pipe_.fit(s)
        self.freq_ = self.pipe_.steps["deseasonal_residual"]["model"].freq_
        self.seasonal_ = self.pipe_.steps["deseasonal_residual"][
            "model"
        ].seasonal_

    def _predict_core(self, s: pd.Series) -> pd.Series:
        self._sync_params()
        return self.pipe_.detect(s)
