"""Module for high-dimensional transformers.

High-dimensional transformers transform hight-dimensional time series, i.e.
pandas DataFrame, into different series, to extract useful information out of
the original time series.

"""

import pandas as pd
from sklearn.decomposition import PCA

from .._transformer_base import _TransformerHD

__all__ = [
    "RegressionResidual",
    "PcaProjection",
    "PcaReconstruction",
    "PcaReconstructionError",
    "SumAll",
    "CustomizedTransformerHD",
]


class CustomizedTransformerHD(_TransformerHD):
    """Transformer derived from a user-given function and parameters.

    Parameters
    ----------
    transform_func: function
        A function transforming given time serie into new one. The first input
        argument must be a pandas Dataframe, optional input argument allows;
        the output must be a pandas Series or DataFrame with the same index as
        input.

    transform_func_params: dict, optional
        Parameters of transform_func. Default: None.

    fit_func: function, optional
        A function learning from a list of time series and return parameters
        dict that transform_func can used for future transformation. Default:
        None.

    fit_func_params: dict, optional
        Parameters of fit_func. Default: None.

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

    def _fit_core(self, df):
        if self.fit_func is not None:
            if self.fit_func_params is not None:
                fit_func_params = self.fit_func_params
            else:
                fit_func_params = {}
            self._fitted_transform_func_params = self.fit_func(
                df, **fit_func_params
            )

    def _predict_core(self, df):
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


class SumAll(_TransformerHD):
    """Transformer that returns the sum all series as one series."""

    _need_fit = False

    def __init__(self):
        super().__init__()

    def _fit_core(self, df):
        pass

    def _predict_core(self, df):
        return df.sum(axis=1, skipna=False)


class RegressionResidual(_TransformerHD):
    """Transformer that performs regression to build relationship between a
    target series and the rest of series, and returns regression residual
    series.

    Parameters
    ----------
    regressor: object
        Regressor to be used. Same as a scikit-learn regressor, it should
        minimally have `fit` and `predict` methods.
    target: str, optional
        Name of the column to be regarded as target variable. If not specified,
        the first column in input DataFrame will be used.

    """

    _need_fit = True
    _default_params = {"regressor": None, "target": None}

    def __init__(
        self,
        regressor=_default_params["regressor"],
        target=_default_params["target"],
    ):
        super().__init__(regressor=regressor, target=target)

    def _fit_core(self, df):
        if self.regressor is None:
            raise RuntimeError("Regressor is not specified.")
        if self.target is None:
            self._target = df.columns[0]
        else:
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

    def _predict_core(self, df):
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
        residual = pd.Series(index=df.index)
        if not df.dropna().empty:
            residual.loc[df.dropna().index] = df.dropna().loc[
                :, target
            ] - self.regressor.predict(df.dropna().loc[:, features])
        return residual


class PcaProjection(_TransformerHD):
    """Transformer that performs principal component analysis (PCA) to the
    multivariate time series (every time point is treated as a point in high-
    dimensional space), and represent those points with their projection on
    the first k principal components.

    Parameters
    ----------
    k: int, optional
        Number of principal components to use. Default: 1.

    """

    _need_fit = True
    _default_params = {"k": 1}

    def __init__(self, k=_default_params["k"]):
        self._model = None
        super().__init__(k=k)

    def _fit_core(self, df):
        self._model = PCA(n_components=self.k)
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        self._model.fit(df.dropna().values)

    def _predict_core(self, df):
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


class PcaReconstruction(_TransformerHD):
    """Transformer that performs principal component analysis (PCA) to the
    multivariate time series  (every time point is treated as a point in high-
    dimensional space), and reconstruct those points with the first k principal
    components.

    Parameters
    ----------
    k: int, optional
        Number of principal components to use. Default: 1.

    """

    _need_fit = True
    _default_params = {"k": 1}

    def __init__(self, k=_default_params["k"]):
        self._model = None
        super().__init__(k=k)

    def _fit_core(self, df):
        self._model = PCA(n_components=self.k)
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        self._model.fit(df.dropna().values)

    def _predict_core(self, df):
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


class PcaReconstructionError(_TransformerHD):
    """Transformer that performs principal component analysis (PCA) to the
    multivariate time series  (every time point is treated as a point in high-
    dimensional space), reconstruct those points with the first k principal
    components, and return the reconstruction error (i.e. squared distance
    bewteen the reconstructed point and original point).

    Parameters
    ----------
    k: int, optional
        Number of principal components to use. Default: 1.

    """

    _need_fit = True
    _default_params = {"k": 1}

    def __init__(self, k=_default_params["k"]):
        self._model = None
        super().__init__(k=k)

    def _fit_core(self, df):
        self._model = PCA(n_components=self.k)
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        self._model.fit(df.dropna().values)

    def _predict_core(self, df):
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
