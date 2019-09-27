"""Module for 1-dimensional transformers.

1-dimensional transformers transform 1-dimensional time series, i.e. pandas
Series, into different series, to extract useful information out of the
original time series.

"""

import warnings

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import acf

from .._transformer_base import _Transformer1D

__all__ = [
    "RollingAggregate",
    "DoubleRollingAggregate",
    "NaiveSeasonalDecomposition",
    "STLDecomposition",
    "Retrospect",
    "StandardScale",
    "CustomizedTransformer1D",
]


class CustomizedTransformer1D(_Transformer1D):
    """Transformer derived from a user-given function and parameters.

    Parameters
    ----------
    transform_func: function
        A function transforming given time serie into new one. The first input
        argument must be a pandas Series, optional input argument allows; the
        output must be a pandas Series or DataFrame with the same index as
        input.

    transform_func_params: dict, optional
        Parameters of transform_func. Default: None.

    fit_func: function, optional
        A function learning from a list of time series and return parameters
        dict that transform_func can used for future transformation. Default:
        None.

    fit_func_params: dict, optional
        Parameters of fit_func. Default: None.


    This is an univariate transformer. When it is applied to a multivariate
    time series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _need_fit = False
    _default_params = {
        "transform_func": None,
        "transform_func_params": None,
        "fit_func": None,
        "fit_func_params": None,
    }

    def __init__(
        self,
        transform_func=_default_params["transform_func"],
        transform_func_params=_default_params["transform_func_params"],
        fit_func=_default_params["fit_func"],
        fit_func_params=_default_params["fit_func_params"],
    ):
        self._fitted_transform_func_params = {}
        super().__init__(
            transform_func=transform_func,
            transform_func_params=transform_func_params,
            fit_func=fit_func,
            fit_func_params=fit_func_params,
        )

    def _fit_core(self, s):
        if self.fit_func is not None:
            if self.fit_func_params is not None:
                fit_func_params = self.fit_func_params
            else:
                fit_func_params = {}
            self._fitted_transform_func_params = self.fit_func(
                s, **fit_func_params
            )

    def _predict_core(self, s):
        if self.transform_func_params is not None:
            transform_func_params = self.transform_func_params
        else:
            transform_func_params = {}
        if self.fit_func is not None:
            return self.transform_func(
                s,
                **{
                    **self._fitted_transform_func_params,
                    **transform_func_params,
                }
            )
        else:
            return self.transform_func(s, **transform_func_params)

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


class StandardScale(_Transformer1D):
    """Transformer that scales time series such that mean is equal to 0 and
    standard deviation is equal to 1.


    This is an univariate transformer. When it is applied to a multivariate
    time series (i.e. pandas DataFrame), it will be applied to every series
    independently.

    """

    _need_fit = False

    def __init__(self):
        super().__init__()

    def _fit_core(self, s):
        pass

    def _predict_core(self, s):
        mean = s.mean()
        std = s.std()

        if std == 0:
            std = 1

        return (s - mean) / std


class RollingAggregate(_Transformer1D):
    """Transformer that roll a sliding window along a time series, and
    aggregates using a user-selected operation.

    Parameters
    ----------
    agg: str or function
        Aggregation method applied to series.
        If str, must be one of supported built-in methods:

        - 'mean': mean of all values in a rolling window.
        - 'median': median of all values in a rolling window.
        - 'sum': summation of all values in a rolling window.
        - 'min': minimum of all values in a rolling window.
        - 'max': maximum of all values in a rolling window.
        - 'std': sample standard deviation of all values in a rolling window.
        - 'var': sample variance of all values in a rolling window.
        - 'skew': skewness of all values in a rolling window.
        - 'kurt': kurtosis of all values in a rolling window.
        - 'count': number of non-nan values in a rolling window.
        - 'nnz': number of non-zero values in a rolling window.
        - 'nunique': number of unique values in a rolling window.
        - 'quantile': quantile of all values in a rolling window. Require
          percentile parameter `q` in in parameter `agg_params`, which is a
          float or a list of float between 0 and 1 inclusive.
        - 'iqr': interquartile range, i.e. difference between 75% and 25%
          quantiles.
        - 'idr': interdecile range, i.e. difference between 90% and 10%
          quantiles.
        - 'hist': histogram of all values in a rolling window. Require
          parameter `bins` in parameter `agg_params` to define the bins. `bins`
          is either a list of floats, b1, ..., bn, which defines n-1 bins
          [b1, b2), [b2, b3), ..., [b{n-2}, b{n-1}), [b{n-1}, bn], or an
          integer that defines the number of equal-width bins in the range of
          input series.

        If function, it should accept a rolling window in form of a pandas
        Series, and return either a scalar or a 1D numpy array. To specify
        names of outputs, specify a list of strings as a parameter `names` in
        parameter `agg_params`.

        Default: 'mean'

    agg_params: dict, optional
        Parameters of aggregation function. Default: None.

    window: int, optional
        Width of rolling windows (number of data points). Default: 10.

    center: bool, optional
        Whether the calculation is at the center of time window or on the right
        edge. Default: False.

    min_periods: int, optional
        Minimum number of observations in window required to have a value.
        Default: None, i.e. all observations must have values.


    This is an univariate transformer. When it is applied to a multivariate
    time series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _need_fit = False
    _default_params = {
        "agg": "mean",
        "agg_params": None,
        "window": 10,
        "center": False,
        "min_periods": None,
    }

    def __init__(
        self,
        agg=_default_params["agg"],
        agg_params=_default_params["agg_params"],
        window=_default_params["window"],
        center=_default_params["center"],
        min_periods=_default_params["min_periods"],
    ):
        super().__init__(
            agg=agg,
            agg_params=agg_params,
            window=window,
            center=center,
            min_periods=min_periods,
        )
        self._closed = None

    def _fit_core(self, s):
        pass

    def _predict_core(self, s):
        if not (
            s.index.is_monotonic_increasing or s.index.is_monotonic_decreasing
        ):
            warnings.warn(
                "Time series does not have a monotonic increasing time index. "
                "Results from this model may be unreliable.",
                UserWarning,
            )

        agg = self.agg
        agg_params = self.agg_params if (self.agg_params is not None) else {}
        window = self.window
        center = self.center
        min_periods = self.min_periods
        closed = self._closed

        rolling = s.rolling(
            window=window,
            center=center,
            min_periods=min_periods,
            closed=closed,
        )

        def getRollingVector(rolling, aggFunc, output_names):
            # we use this function to trick pandas to get vector rolling agg
            s_rolling_raw = []

            def agg_wrapped(x):
                s_rolling_raw.append(aggFunc(x))
                return 0

            s_rolling = rolling.agg(agg_wrapped)
            s_rolling_raw = np.array(s_rolling_raw)
            if s_rolling_raw.ndim == 1:
                s_rolling[s_rolling.notna()] = np.array(s_rolling_raw)
            elif s_rolling_raw.shape[1] <= 1:
                s_rolling[s_rolling.notna()] = np.array(s_rolling_raw)
            else:
                df = pd.concat([s_rolling] * s_rolling_raw.shape[1], axis=1)
                df[s_rolling.notna()] = s_rolling_raw
                s_rolling = df
            if output_names is not None:
                if isinstance(s_rolling, pd.Series):
                    s_rolling.name = output_names
                if isinstance(s_rolling, pd.DataFrame):
                    s_rolling.columns = output_names
            return s_rolling

        aggList = [
            "mean",
            "median",
            "sum",
            "min",
            "max",
            "quantile",
            "iqr",
            "idr",
            "count",
            "nnz",
            "nunique",
            "std",
            "var",
            "skew",
            "kurt",
            "hist",
        ]
        if agg in [
            "mean",
            "median",
            "sum",
            "min",
            "max",
            "count",
            "std",
            "var",
            "skew",
            "kurt",
        ]:
            s_rolling = rolling.agg(agg)
        elif agg == "nunique":
            s_rolling = rolling.agg(lambda x: len(np.unique(x.dropna())))
        elif agg == "nnz":
            s_rolling = rolling.agg(np.count_nonzero)
        elif agg == "quantile":
            if hasattr(agg_params["q"], "__iter__"):
                s_rolling = pd.concat(
                    [
                        rolling.quantile(q).rename("q{}".format(q))
                        for q in agg_params["q"]
                    ],
                    axis=1,
                )
            else:
                s_rolling = rolling.quantile(agg_params["q"])
        elif agg == "iqr":
            s_rolling = rolling.quantile(0.75) - rolling.quantile(0.25)
        elif agg == "idr":
            s_rolling = rolling.quantile(0.9) - rolling.quantile(0.1)
        elif agg == "hist":
            if isinstance(agg_params["bins"], int):
                bins = np.histogram_bin_edges(
                    s.dropna().values, bins=agg_params["bins"]
                )
            else:
                bins = agg_params["bins"]
            s_rolling = getRollingVector(
                rolling,
                lambda x: np.histogram(x, bins=bins)[0],
                (
                    [
                        "[{}, {}{}".format(
                            bins[i],
                            bins[i + 1],
                            ")" if i < len(bins) - 2 else "]",
                        )
                        for i in range(len(bins) - 1)
                    ]
                ),
            )
        elif callable(agg):
            try:
                s_rolling = rolling.agg(agg)
            except TypeError:
                if "names" in agg_params.keys():
                    s_rolling = getRollingVector(
                        rolling, agg, agg_params["names"]
                    )
                else:
                    s_rolling = getRollingVector(rolling, agg, None)
        else:
            raise ValueError("Attribute agg must be one of {}".format(aggList))

        if isinstance(s_rolling, pd.Series):
            s_rolling.name = s.name
        else:
            if s.name is not None:
                s_rolling.columns = [
                    "{}_{}".format(s.name, col) for col in s_rolling.columns
                ]
        return s_rolling


class DoubleRollingAggregate(_Transformer1D):
    """Transformer that rolls two sliding windows side-by-side along a time
    series, aggregates using a user-given operation, and calcuates the
    difference of aggregated metrics between two sliding windows.

    Parameters
    ----------
    agg: str, function, or tuple
        Aggregation method applied to series.
        If str, must be one of supported built-in methods:

        - 'mean': mean of all values in a rolling window.
        - 'median': median of all values in a rolling window.
        - 'sum': summation of all values in a rolling window.
        - 'min': minimum of all values in a rolling window.
        - 'max': maximum of all values in a rolling window.
        - 'std': sample standard deviation of all values in a rolling window.
        - 'var': sample variance of all values in a rolling window.
        - 'skew': skewness of all values in a rolling window.
        - 'kurt': kurtosis of all values in a rolling window.
        - 'count': number of non-nan values in a rolling window.
        - 'nnz': number of non-zero values in a rolling window.
        - 'nunique': number of unique values in a rolling window.
        - 'quantile': quantile of all values in a rolling window. Require
          percentile parameter `q` in in parameter `agg_params`, which is a
          float or a list of float between 0 and 1 inclusive.
        - 'iqr': interquartile range, i.e. difference between 75% and 25%
          quantiles.
        - 'idr': interdecile range, i.e. difference between 90% and 10%
          quantiles.
        - 'hist': histogram of all values in a rolling window. Require
          parameter `bins` in parameter `agg_params` to define the bins. `bins`
          is either a list of floats, b1, ..., bn, which defines n-1 bins
          [b1, b2), [b2, b3), ..., [b{n-2}, b{n-1}), [b{n-1}, bn], or an
          integer that defines the number of equal-width bins in the range of
          input series.

        If function, it should accept a rolling window in form of a pandas
        Series, and return either a scalar or a 1D numpy array. To specify
        names of outputs, specify a list of strings as a parameter `names` in
        parameter `agg_params`.

        If tuple, elements correspond left and right window respectively.

        Default: 'mean'

    agg_params: dict or tuple, optional
        Parameters of aggregation function. If tuple, elements correspond left
        and right window respectively. Default: None.

    window: int or tuple, optional
        Width of rolling windows (number of data points). If tuple, elements
        correspond left and right window respectively. Default: 10.

    center: bool, optional
        If True, the current point is the right edge of right window;
        Otherwise, it is the right edge of left window.
        Default: True.

    min_periods: int or tuple, optional
        Minimum number of observations in window required to have a value.
        Default: None, i.e. all observations must have values.

    diff: str or function, optional
        Difference method applied between aggregated metrics from the two
        sliding windows.
        If str, choose from supported built-in methods:

        - 'diff': Difference between values of aggregated metric (right minus
          left). Only applicable if the aggregated metric is scalar.
        - 'rel_diff': Relative difference between values of aggregated metric
          (right minus left divided left). Only applicable if the aggregated
          metric is scalar.
        - 'abs_rel_diff': Absolute relative difference between values of
          aggregated metric (right minus left divided left). Only applicable if
          the aggregated metric is scalar.
        - 'l1': Absolute difference if aggregated metric is scalar, or sum of
          elementwise absolute difference if it is a vector.
        - 'l2': Square root of sum of elementwise squared difference.

        If function, it accepts two input arguments that are the two outputs of
        applying aggregation method to the two windows, and returns a float
        number measuring the difference.

        Default: 'l1'


    This is an univariate transformer. When it is applied to a multivariate
    time series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _need_fit = False
    _default_params = {
        "agg": "mean",
        "agg_params": None,
        "window": 10,
        "center": True,
        "min_periods": None,
        "diff": "l1",
    }

    def __init__(
        self,
        agg=_default_params["agg"],
        agg_params=_default_params["agg_params"],
        window=_default_params["window"],
        center=_default_params["center"],
        min_periods=_default_params["min_periods"],
        diff=_default_params["diff"],
    ):
        super().__init__(
            agg=agg,
            agg_params=agg_params,
            window=window,
            min_periods=min_periods,
            center=center,
            diff=diff,
        )

    def _fit_core(self, s):
        pass

    def _predict_core(self, s):
        if not (
            s.index.is_monotonic_increasing or s.index.is_monotonic_decreasing
        ):
            warnings.warn(
                "Time series does not have a monotonic increasing time index. "
                "Results from this model may be unreliable.",
                UserWarning,
            )

        agg = self.agg
        agg_params = self.agg_params if (self.agg_params is not None) else {}
        window = self.window
        min_periods = self.min_periods
        center = self.center
        diff = self.diff

        if not isinstance(agg, tuple):
            agg = (agg, agg)

        if not isinstance(agg_params, tuple):
            agg_params = (agg_params, agg_params)

        if not isinstance(window, tuple):
            window = (window, window)

        if not isinstance(min_periods, tuple):
            min_periods = (min_periods, min_periods)

        if center:
            if isinstance(window[0], int):
                s_rolling_left = RollingAggregate(
                    agg=agg[0],
                    agg_params=agg_params[0],
                    window=window[0],
                    min_periods=min_periods[0],
                    center=False,
                ).transform(s.shift(1))
            else:
                ra = RollingAggregate(
                    agg=agg[0],
                    agg_params=agg_params[0],
                    window=window[0],
                    min_periods=min_periods[0],
                    center=False,
                )
                ra._closed = "left"
                s_rolling_left = ra.transform(s)
            if isinstance(window[1], int):
                s_rolling_right = (
                    RollingAggregate(
                        agg=agg[1],
                        agg_params=agg_params[1],
                        window=window[1],
                        min_periods=min_periods[1],
                        center=False,
                    )
                    .transform(s.iloc[::-1])
                    .iloc[::-1]
                )
            else:
                s_reversed = pd.Series(
                    s.values[::-1],
                    index=pd.DatetimeIndex(
                        [
                            s.index[0] + (s.index[-1] - s.index[i])
                            for i in range(len(s) - 1, -1, -1)
                        ]
                    ),
                )
                s_rolling_right = pd.Series(
                    RollingAggregate(
                        agg=agg[1],
                        agg_params=agg_params[1],
                        window=window[1],
                        min_periods=min_periods[1],
                        center=False,
                    )
                    .transform(s_reversed)
                    .iloc[::-1]
                    .values,
                    index=s.index,
                )
                s_rolling_right.name = s.name
        else:
            if isinstance(window[1], int):
                s_rolling_left = RollingAggregate(
                    agg=agg[0],
                    agg_params=agg_params[0],
                    window=window[0],
                    min_periods=min_periods[0],
                    center=False,
                ).transform(s.shift(window[1]))
            else:
                s_shifted = pd.Series(
                    s.values, s.index + pd.Timedelta(window[1])
                )
                s_shifted = s_shifted.append(pd.Series(index=s.index))
                s_shifted = s_shifted.iloc[
                    s_shifted.index.duplicated() == False
                ]
                s_shifted = s_shifted.sort_index()
                s_shifted.name = s.name
                s_rolling_left = RollingAggregate(
                    agg=agg[0],
                    agg_params=agg_params[0],
                    window=window[0],
                    min_periods=min_periods[0],
                    center=False,
                ).transform(s_shifted)
                if isinstance(s_rolling_left, pd.Series):
                    s_rolling_left = s_rolling_left[s.index]
                else:
                    s_rolling_left = s_rolling_left.loc[s.index, :]
            s_rolling_right = RollingAggregate(
                agg=agg[1],
                agg_params=agg_params[1],
                window=window[1],
                min_periods=min_periods[1],
                center=False,
            ).transform(s)

        if isinstance(s_rolling_left, pd.Series):
            if diff in ["l1", "l2"]:
                return abs(s_rolling_right - s_rolling_left)
            if diff == "diff":
                return s_rolling_right - s_rolling_left
            if diff == "rel_diff":
                return (s_rolling_right - s_rolling_left) / s_rolling_left
            if diff == "abs_rel_diff":
                return abs(s_rolling_right - s_rolling_left) / s_rolling_left

        if isinstance(s_rolling_left, pd.DataFrame):
            if diff == "l1":
                return abs(s_rolling_right - s_rolling_left).sum(
                    axis=1, skipna=False
                )
            if diff == "l2":
                return ((s_rolling_right - s_rolling_left) ** 2).sum(
                    axis=1, skipna=False
                ) ** 0.5

        if callable(diff):
            s_rolling = s.copy()
            for i in range(len(s_rolling)):
                s_rolling.iloc[i] = diff(
                    s_rolling_left.iloc[i], s_rolling_right.iloc[i]
                )
            return s_rolling

        raise ValueError("Invalid value of diff")


class _Deseasonal(_Transformer1D):
    _default_params = {"freq": None}

    def __init__(self, freq=_default_params["freq"]):
        super().__init__(freq=freq)

    @staticmethod
    def identify_seasonal_period(s, low_autocorr=0.1, high_autocorr=0.3):
        """Identify seasonal period of a time series based on autocorrelation.

        Parameters
        ----------
        s: pandas Series or DataFrame
            Time series where to identify seasonal periods.

        low_autocorr: float, optional
            Threshold below which values of autocorreltion are consider low.
            Default: 0.1

        high_autocorr: float, optional
            Threshold below which values of autocorreltion are consider high.
            Default: 0.3

        Returns
        -------
        int
            Seasonal period of the time series. If no significant seasonality
            is found, return None.

        """

        if low_autocorr > high_autocorr:
            raise ValueError("`low_autocorr` must not exceed `high_autocorr`")

        # check if the time series has uniform time step
        if len(np.unique(np.diff(s.index))) > 1:
            raise ValueError("The time steps are not constant. ")

        autocorr = acf(s, nlags=len(s), fft=False)
        cutPos = np.argwhere(autocorr >= low_autocorr)[0][0]
        diff_autocorr = np.diff(autocorr[cutPos:])
        high_autocorr_peak_pos = (
            cutPos
            + 1
            + np.argwhere(
                (diff_autocorr[:-1] > 0)
                & (diff_autocorr[1:] < 0)
                & (autocorr[cutPos + 1 : -1] > high_autocorr)
            ).flatten()
        )
        if len(high_autocorr_peak_pos) > 0:
            return high_autocorr_peak_pos[
                np.argmax(autocorr[high_autocorr_peak_pos])
            ]
        else:
            return None

    def _fit_core(self, s):
        if not (
            s.index.is_monotonic_increasing or s.index.is_monotonic_decreasing
        ):
            warnings.warn(
                "Time series does not have a monotonic increasing time index. "
                "Results from this model may be unreliable.",
                UserWarning,
            )
        # remove starting and ending nans
        s = s.loc[s.first_valid_index() : s[::-1].first_valid_index()].copy()
        if pd.isna(s).any():
            raise ValueError(
                "Found NaN in time series among valid values. "
                "NaNs starting or ending a time series are allowed, "
                "but those among valid values are not."
            )
        # get datum time
        self._datumTimestamp = s.index[0]
        # get series_freq
        if s.index.freq is not None:
            self._series_freq = s.index.freqstr
        else:
            self._series_freq = s.index.inferred_freq
        if self._series_freq is None:
            raise RuntimeError(
                "Series does not follow any known frequency "
                "(e.g. second, minute, hour, day, week, month, year, etc."
            )
        # get average dT
        self._dT = pd.Series(s.index).diff().mean()
        # get seasonal freq
        self._get_seasonal_freq(s)
        # get seasonal pattern
        if self.freq_ > len(s):
            raise RuntimeError("Training series is shorter than frequency.")
        self._get_seasonal_pattern(s)

    def _predict_core(self, s):
        if not (
            s.index.is_monotonic_increasing or s.index.is_monotonic_decreasing
        ):
            warnings.warn(
                "Time series does not have a monotonic increasing time index. "
                "Results from this model may be unreliable.",
                UserWarning,
            )
        # check if series freq is same
        if self._series_freq not in {s.index.freqstr, s.index.inferred_freq}:
            raise RuntimeError(
                "Model was trained by a series whose index has {} frequency, "
                "but is tranforming a series whose index has {} frequency.".format(
                    self._series_freq, s.index.freq
                )
            )
        # get phase shift
        approx_steps = (s.index[0] - self._datumTimestamp) / self._dT
        # try to find starting_phase
        if approx_steps > 0:
            helper_index = pd.date_range(
                start=self._datumTimestamp,
                periods=round(approx_steps) + 100,
                freq=self._series_freq,
            )
            if helper_index[-1] <= s.index[0]:
                raise RuntimeError("You shouldn't have reached here...")
            for i in range(len(helper_index) - 1, -1, -1):
                if helper_index[i] == s.index[0]:
                    starting_phase = i % self.freq_
                    break
                elif helper_index[i] < s.index[0]:
                    raise RuntimeError(
                        "The series to be transformed has different "
                        "phases from the series used to train the model."
                    )
                else:
                    pass
            else:
                raise RuntimeError(
                    "You definitely shouldn't have reached here..."
                )
        elif approx_steps < 0:
            helper_index = pd.date_range(
                end=self._datumTimestamp,
                periods=round(-approx_steps) + 100,
                freq=self._series_freq,
            )
            if helper_index[0] >= s.index[0]:
                raise RuntimeError("You shouldn't have reached here...")
            for i in range(len(helper_index)):
                if helper_index[i] == s.index[0]:
                    starting_phase = (len(helper_index) - 1 - i) % self.freq_
                    if starting_phase != 0:
                        starting_phase = self.freq_ - starting_phase
                    break
                elif helper_index[i] > s.index[0]:
                    raise RuntimeError(
                        "The series to be transformed has different "
                        "phases from the series used to train the model."
                    )
                else:
                    pass
            else:
                raise RuntimeError(
                    "You definitely shouldn't have reached here..."
                )
        else:
            starting_phase = 0
        # get seasonal series and remove it from original
        phase_pattern = np.concatenate(
            [np.arange(starting_phase, self.freq_), np.arange(starting_phase)]
        )
        s_seasonal = pd.Series(
            self.seasonal_.values[
                phase_pattern[np.arange(len(s)) % self.freq_]
            ],
            index=s.index,
        )
        s_residual = s - s_seasonal
        s_residual = self._remove_trend(s_residual)
        return s_residual


class NaiveSeasonalDecomposition(_Deseasonal):
    """Transformer that performs naive seasonal decomposition to the time
    series, and returns residual series.

    Naive seasonal decomposition assumes time series is seasonal series plus
    noise (residual). This transformer obtains seasonal pattern by taking
    average over seasonal periods, removes the seasonal part and returns
    residual series. It may not applicable for seasonal time series with strong
    trend besides seasonal effects, where STLDecomposition may be helpful.

    Parameters
    ----------
    freq: int, optional
        Length of a seasonal cycle. If None, the model will determine based on
        autocorrelation of the training series. Default: None.

    Attributes
    ----------
    freq_: int
        Length of seasonal cycle. Equal to parameter `freq` if it is given.
        Otherwise, calculated based on autocorrelation of the training series.

    seasonal_: pandas.Series
        Seasonal pattern extracted from training series.


    This is an univariate transformer. When it is applied to a multivariate
    time series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    def _get_seasonal_freq(self, s):
        if self.freq is None:
            self.freq_ = self.identify_seasonal_period(s)
            if self.freq_ is None:
                raise Exception("Could not find significant seasonality.")
        else:
            self.freq_ = self.freq

    def _get_seasonal_pattern(self, s):
        self.seasonal_ = s.iloc[: self.freq_].copy()
        for i in range(len(self.seasonal_)):
            self.seasonal_.iloc[i] = s.iloc[i :: len(self.seasonal_)].mean()

    def _remove_trend(self, s):
        return s


class STLDecomposition(_Deseasonal):
    """Transformer that performs STL decomposition ("Seasonal and Trend
    decomposition using Loess") to the time series, and returns residual
    series.

    STL decomposition assumes time series is sum of trend, seasonal pattern and
    noise (residual). This transformer performs STL decomposition to a time
    series and returns residual series.

    Parameters
    ----------
    freq: int, optional
        Length of a seasonal cycle. If None, the model will determine based on
        autocorrelation of the time series. Default: None.

    Attributes
    ----------
    freq_: int
        Length of seasonal cycle. Equal to parameter `freq` if it is given.
        Otherwise, calculated based on autocorrelation of the training series.

    seasonal_: pandas.Series
        Seasonal pattern extracted from training series.


    This is an univariate transformer. When it is applied to a multivariate
    time series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    def _get_seasonal_freq(self, s):
        if self.freq is None:
            self.freq_ = self.identify_seasonal_period(s)
            if self.freq_ is None:
                self.freq_ = 2
            MAX_ITER = 20
            for _ in range(MAX_ITER):
                s_trend = s.rolling(
                    window=self.freq_, center=True, min_periods=1
                ).mean()
                s_detrended = s - s_trend
                freq_d = self.identify_seasonal_period(s_detrended)
                if freq_d is None:
                    freq_d = int(self.freq_ * 2)
                if freq_d == self.freq_:
                    break
                else:
                    self.freq_ = freq_d
            else:
                raise Exception("Could not find significant seasonality.")
        else:
            self.freq_ = self.freq

    def _get_seasonal_pattern(self, s):
        result = seasonal_decompose(s, freq=self.freq_)
        self.seasonal_ = getattr(result, "seasonal")[: self.freq_]

    def _remove_trend(self, s):
        s_trend = s.rolling(
            window=(self.freq_ if self.freq_ % 2 else self.freq_ + 1),
            center=True,
        ).mean()
        return s - s_trend


class Retrospect(_Transformer1D):
    """Transformer that returns dataframe with retrospective values, i.e. a row
    at time t includes value at (t-k)'s where k's are specified by user.

    This transformer may be useful for cases where lagging effect should be
    taken in account. For example, a change of control u may not be reflected
    in outcome y within 2 minutes, and its effect may last for another 3
    minutes. In this case, a dataframe where each row include u_[t-3], u_[t-4],
    u_[t-5], and a series y_t are needed to learn the relationship between
    control and outcome.

    Parameters
    ----------
    n_steps: int, optional
        Number of retrospective steps to take. Default: 1.
    step_size: int, optional
        Length of a retrospective step. Default: 1.
    till: int, optional
        Nearest retrospective step. Default: 0.

    Examples
    --------
    >>> s = pd.Series(
            np.arange(10),
            index=pd.date_range(
                start='2017-1-1',
                periods=10,
                freq='D'))
            2017-01-01    0
            2017-01-02    1
            2017-01-03    2
            2017-01-04    3
            2017-01-05    4
            2017-01-06    5
            2017-01-07    6
            2017-01-08    7
            2017-01-09    8
            2017-01-10    9
    >>> Retrospect(n_steps=3, step_size=2, till=1).transform(s)
                        t-1	t-3	t-5
            2017-01-01	NaN	NaN	NaN
            2017-01-02	0.0	NaN	NaN
            2017-01-03	1.0	NaN	NaN
            2017-01-04	2.0	0.0	NaN
            2017-01-05	3.0	1.0	NaN
            2017-01-06	4.0	2.0	0.0
            2017-01-07	5.0	3.0	1.0
            2017-01-08	6.0	4.0	2.0
            2017-01-09	7.0	5.0	3.0
            2017-01-10	8.0	6.0	4.0


    This is an univariate transformer. When it is applied to a multivariate
    time series (i.e. pandas DataFrame), it will be applied to every series
    independently. All parameters can be defined as a dict object where key-
    value pairs are series names (i.e. column names of DataFrame) and the
    model parameter for that series. If not, then the same parameter will be
    applied to all series.

    """

    _need_fit = False
    _default_params = {"n_steps": 1, "step_size": 1, "till": 0}

    def __init__(
        self,
        n_steps=_default_params["n_steps"],
        step_size=_default_params["step_size"],
        till=_default_params["till"],
    ):
        super().__init__(n_steps=n_steps, step_size=step_size, till=till)

    def _fit_core(self, s):
        pass

    def _predict_core(self, s):
        if not (
            s.index.is_monotonic_increasing or s.index.is_monotonic_decreasing
        ):
            warnings.warn(
                "Time series does not have a monotonic increasing time index. "
                "Results from this model may be unreliable.",
                UserWarning,
            )
        n_steps = self.n_steps
        till = self.till
        step_size = self.step_size
        df = pd.DataFrame(index=s.index)
        df = df.assign(
            **{
                (
                    "t-{}".format(i)
                    if s.name is None
                    else "{}_t-{}".format(s.name, i)
                ): s.shift(i)
                for i in range(till, till + n_steps * step_size, step_size)
            }
        )
        return df
