"""Module for high-dimensional transformers.

High-dimensional transformers transform hight-dimensional time series, i.e.
pandas DataFrame, into different series, to extract useful information out of
the original time series.

"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
from sklearn.decomposition import PCA

from .._transformer_base import (
    _NonTrainableMultivariateTransformer,
    _TrainableMultivariateTransformer,
)


class CustomizedTransformerHD(_TrainableMultivariateTransformer):
    """Multivariate transformer derived from a user-given function and parameters.

    Parameters
    ----------
    Parameters
    ----------
    transform_func: function
        A function transforming multivariate time series.

        The first input argument must be a pandas DataFrame, optional input
        argument may be accepted through parameter `transform_func_params` and
        the output of `fit_func`, and the output must be a pandas Series or
        DataFrame with the same index as input.

    transform_func_params: dict, optional
        Parameters of `transform_func`. Default: None.

    fit_func: function, optional
        A function training parameters of `transform_func` with multivariate
        time series.

        The first input argument must be a pandas DataFrame, optional input
        argument may be accepted through parameter `fit_func_params`, and the
        output must be a dict that can be used by `transform_func` as
        parameters. Default: None.

    fit_func_params: dict, optional
        Parameters of `fit_func`. Default: None.

    """

    def __init__(
        self,
        transform_func: Callable,
        transform_func_params: Optional[Dict[str, Any]] = None,
        fit_func: Optional[Callable] = None,
        fit_func_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        self._fitted_transform_func_params = {}  # type: Dict
        super().__init__()
        self.transform_func = transform_func
        self.transform_func_params = transform_func_params
        self.fit_func = fit_func
        self.fit_func_params = fit_func_params
        if self.fit_func is None:
            self._fitted = 1

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return (
            "transform_func",
            "transform_func_params",
            "fit_func",
            "fit_func_params",
        )

    def _fit_core(self, df: pd.DataFrame) -> None:
        if self.fit_func is not None:
            if self.fit_func_params is not None:
                fit_func_params = self.fit_func_params
            else:
                fit_func_params = {}
            self._fitted_transform_func_params = self.fit_func(
                df, **fit_func_params
            )

    def _predict_core(
        self, df: pd.DataFrame
    ) -> Union[pd.Series, pd.DataFrame]:
        if self.transform_func_params is not None:
            transform_func_params = self.transform_func_params
        else:
            transform_func_params = {}
        if self.fit_func is not None:
            return self.transform_func(
                df,
                **{
                    **self._fitted_transform_func_params,
                    **transform_func_params,
                }
            )
        else:
            return self.transform_func(df, **transform_func_params)


class SumAll(_NonTrainableMultivariateTransformer):
    """Transformer that returns the sum all series as one series."""

    def __init__(self) -> None:
        super().__init__()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return tuple()

    def _predict_core(self, df: pd.DataFrame) -> pd.Series:
        return df.sum(axis=1, skipna=False)


class RegressionResidual(_TrainableMultivariateTransformer):
    """Transformer that performs regression to build relationship between a
    target series and the rest of series, and returns regression residual
    series.

    Parameters
    ----------
    regressor: object
        Regressor to be used. Same as a scikit-learn regressor, it should
        minimally have `fit` and `predict` methods.
    target: str, optional
        Name of the column to be regarded as target variable.

    """

    def __init__(self, regressor: Any, target: str) -> None:
        super().__init__()
        self.regressor = regressor
        self.target = target

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("regressor", "target")

    def _fit_core(self, df: pd.DataFrame) -> None:
        if self.target not in df.columns:
            raise RuntimeError(
                "Cannot find target series {} in input dataframe.".format(
                    self.target
                )
            )
        self._target = self.target
        self._features = [col for col in df.columns if col != self._target]
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        self.regressor.fit(
            df.dropna().loc[:, self._features],
            df.dropna().loc[:, self._target],
        )

    def _predict_core(self, df: pd.DataFrame) -> pd.Series:
        target = self._target
        features = self._features
        if target not in df.columns:
            raise RuntimeError(
                "Cannot find target series {} in input dataframe.".format(
                    target
                )
            )
        if not set(features) <= set(df.columns):
            raise RuntimeError(
                "The following series are not found in input dataframe: {}.".format(
                    set(features) - set(df.columns)
                )
            )
        residual = pd.Series(index=df.index, dtype="float64")
        if not df.dropna().empty:
            residual.loc[df.dropna().index] = df.dropna().loc[
                :, target
            ] - self.regressor.predict(df.dropna().loc[:, features])
        return residual


class PcaProjection(_TrainableMultivariateTransformer):
    """Transformer that performs principal component analysis (PCA) to the
    multivariate time series (every time point is treated as a point in high-
    dimensional space), and represents those points with their projection on
    the first k principal components.

    Parameters
    ----------
    k: int, optional
        Number of principal components to use. Default: 1.

    """

    def __init__(self, k: int = 1) -> None:
        self._model = None  # type: Any
        super().__init__()
        self.k = k

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("k",)

    def _fit_core(self, df: pd.DataFrame) -> None:
        self._model = PCA(n_components=self.k)
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        self._model.fit(df.dropna().values)

    def _predict_core(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.k > self._model.n_components:
            raise ValueError(
                "k is increased after previous fitting. Please fit again."
            )
        results = pd.DataFrame(
            index=df.index, columns=["pc{}".format(i) for i in range(self.k)]
        )
        if not df.dropna().empty:
            results.loc[df.dropna().index] = self._model.transform(
                df.dropna().values
            )[:, : self.k]
        return results


class PcaReconstruction(_TrainableMultivariateTransformer):
    """Transformer that performs principal component analysis (PCA) to the
    multivariate time series  (every time point is treated as a point in high-
    dimensional space), and reconstructs those points with the first k
    principal components.

    Parameters
    ----------
    k: int, optional
        Number of principal components to use. Default: 1.

    """

    def __init__(self, k: int = 1) -> None:
        self._model = None  # type: Any
        super().__init__()
        self.k = k

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("k",)

    def _fit_core(self, df: pd.DataFrame) -> None:
        self._model = PCA(n_components=self.k)
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        self._model.fit(df.dropna().values)

    def _predict_core(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._model is None:
            raise RuntimeError("Please fit the model first.")
        if self.k > self._model.n_components:
            raise ValueError(
                "k is increased after previous fitting. Please fit again."
            )
        results = pd.DataFrame(columns=df.columns, index=df.index)
        if not df.dropna().empty:
            results.loc[df.dropna().index] = self._model.inverse_transform(
                self._model.transform(df.dropna().values)
            )
        return results


class PcaReconstructionError(_TrainableMultivariateTransformer):
    """Transformer that performs principal component analysis (PCA) to the
    multivariate time series  (every time point is treated as a point in high-
    dimensional space), reconstruct those points with the first k principal
    components, and returns the reconstruction error (i.e. squared distance
    bewteen the reconstructed point and original point).

    Parameters
    ----------
    k: int, optional
        Number of principal components to use. Default: 1.

    """

    def __init__(self, k: int = 1) -> None:
        self._model = None  # type: Any
        super().__init__()
        self.k = k

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("k",)

    def _fit_core(self, df: pd.DataFrame) -> None:
        self._model = PCA(n_components=self.k)
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        self._model.fit(df.dropna().values)

    def _predict_core(self, df: pd.DataFrame) -> pd.Series:
        if self._model is None:
            raise RuntimeError("Please fit the model first.")
        if self.k > self._model.n_components:
            raise ValueError(
                "k is increased after previous fitting. Please fit again."
            )
        results = pd.DataFrame(columns=df.columns, index=df.index)
        if not df.dropna().empty:
            results.loc[df.dropna().index] = self._model.inverse_transform(
                self._model.transform(df.dropna().values)
            )
        return ((results - df) ** 2).sum(axis=1, skipna=False)


class RollingCrossCorrelation(_NonTrainableMultivariateTransformer):
    """Transformer that rolls a sliding window along a multivariate time series
    and returns moving cross correlation.

    Parameters
    ----------
    window: int or str
        Size of the rolling time window.

        - If int, it is the number of time point in this time window.
        - If str, it must be able to be converted into a pandas Timedelta
          object.

    pairs : tuple or list, optional
        Pairs of series to calculate cross correlation.

        - If 2-tuple, return the cross correlation of these two series.
        - If list of tuples, return the cross correlation of every pair in the
          list.
        - If None, return the cross correlation of all possible pairs.

        Default: None.

    center: bool, optional
        Whether the calculation is at the center of time window or on the right
        edge. Default: False.

    min_periods: int, optional
        Minimum number of observations in window required to have a value.
        Default: None, i.e. all observations must have values.

    """

    def __init__(
        self,
        window: Union[int, str],
        pairs: Optional[Union[Tuple[str, str], List[Tuple[str, str]]]] = None,
        center: bool = False,
        min_periods: Optional[int] = None,
    ):
        super().__init__()
        self.window = window
        self.pairs = pairs
        self.center = center
        self.min_periods = min_periods

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("window", "pairs", "center", "min_periods")

    def _predict_core(
        self, df: pd.DataFrame
    ) -> Union[pd.Series, pd.DataFrame]:
        if len(df.columns) <= 1:
            raise ValueError(
                "The input data frame must contain at least two series."
            )

        if self.pairs is None:
            pairs = [
                (df.columns[i], df.columns[j])
                for i in range(len(df.columns))
                for j in range(i + 1, len(df.columns))
            ]
        elif isinstance(self.pairs, tuple):
            pairs = [self.pairs]
        else:
            pairs = self.pairs

        columns = list(
            set(sum([[i, j] for (i, j) in pairs], []))
        )  # all columns in the list of pairs

        if not set(columns) <= set(df.columns):
            raise ValueError(
                "Parameter `pairs` contains a column that is not included in "
                "the data frame."
            )

        rolling_corr = (
            df.loc[:, columns]
            .rolling(
                window=self.window,
                center=self.center,
                min_periods=self.min_periods,
            )
            .corr()
        )

        rolling_corr = pd.DataFrame(
            {
                "{}:{}".format(col_0, col_1): rolling_corr.loc[:, col_0].loc[
                    :, col_1
                ]
                for col_0, col_1 in pairs
            }
        )

        if (len(pairs) == 1) and (
            (self.pairs is None) or isinstance(self.pairs, tuple)
        ):
            rolling_corr = rolling_corr.iloc[:, 0].rename(None)

        return rolling_corr
