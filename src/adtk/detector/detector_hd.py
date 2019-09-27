"""Module for high-dimensional detectors.

High-dimensional detectors detect anomalies from high-dimensional time series,
i.e. from pandas DataFrame.
"""

from collections import Counter

import pandas as pd

from ..aggregator import AndAggregator
from .._detector_base import _DetectorHD
from ..detector import InterQuartileRangeAD, ThresholdAD
from ..pipe import Pipenet, Pipeline
from ..transformer import (
    CustomizedTransformer1D,
    RegressionResidual,
    PcaReconstructionError,
)

__all__ = [
    "MinClusterDetector",
    "OutlierDetector",
    "RegressionAD",
    "PcaAD",
    "CustomizedDetectorHD",
]


class CustomizedDetectorHD(_DetectorHD):
    """Detector derived from a user-given function and parameters.

    Parameters
    ----------
    detect_func: function
        A function detecting anomalies from given time series. The first input
        argument must be a pandas Dataframe, optional input argument allows;
        the output must be a binary pandas Series with the same index as input.

    detect_func_params: dict, optional
        Parameters of detect_func. Default: None.

    fit_func: function, optional
        A function learning from a list of time series and return parameters
        dict that detect_func can used for future detection. Default: None.

    fit_func_params: dict, optional
        Parameters of fit_func. Default: None.

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
        super().__init__(
            detect_func=detect_func,
            detect_func_params=detect_func_params,
            fit_func=fit_func,
            fit_func_params=fit_func_params,
        )

    def _fit_core(self, df):
        if self.fit_func is not None:
            if self.fit_func_params is not None:
                fit_func_params = self.fit_func_params
            else:
                fit_func_params = {}
            self._fitted_detect_func_params = self.fit_func(
                df, **fit_func_params
            )

    def _predict_core(self, df):
        if self.detect_func_params is not None:
            detect_func_params = self.detect_func_params
        else:
            detect_func_params = {}
        if self.fit_func is not None:
            return self.detect_func(
                df, **{**self._fitted_detect_func_params, **detect_func_params}
            )
        else:
            return self.detect_func(df, **detect_func_params)

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


class MinClusterDetector(_DetectorHD):
    """Detector that detect anomaly based on clustering of historical data.

    This detector peforms clustering using a clustering model, and identifies
    a time points as anomalous if it belongs to the minimal cluster.

    Parameters
    ----------
    model: object
        A clustering model to be used for clustering time series values. Same
        as a clustering model in scikit-learn, the model should minimally have
        a `fit` method and a `predict` method. The `predict` method should
        return an array of cluster labels.

    """

    _default_params = {"model": None}

    def __init__(self, model=_default_params["model"]):
        super().__init__(model=model)

    def _fit_core(self, df):
        if self.model is None:
            raise RuntimeError("Model is not specified.")
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        clustering_result = self.model.fit_predict(df.dropna())
        cluster_count = Counter(clustering_result)
        self._anomalous_cluster_id = cluster_count.most_common()[-1][0]

    def _predict_core(self, df):
        cluster_id = pd.Series(index=df.index)
        if not df.dropna().empty:
            cluster_id.loc[df.dropna().index] = self.model.predict(df.dropna())
        predicted = pd.Series(
            cluster_id == self._anomalous_cluster_id, index=df.index
        )
        predicted[cluster_id.isna()] = float("nan")
        return predicted


class OutlierDetector(_DetectorHD):
    """Detector that detect anomaly based on a outlier detection model.

    This detector peforms time-independent outlier detection using given model,
    and identifies a time points as anomalous if it is labelled as an outlier.

    Parameters
    ----------
    model: object
        An outlier detection model to be used. Same as a outlier detection
        model in scikit-learn (e.g. EllipticEnvelope, IsolationForest,
        LocalOutlierFactor), the model should minimally have a `fit_predict`
        method, or `fit` and `predict` methods. The `fit_predict` or `predict`
        method should return an array of outlier indicators where outliers are
        marked by -1.

    """

    _default_params = {"model": None}

    def __init__(self, model=_default_params["model"]):
        super().__init__(model=model)

    def _fit_core(self, df):
        if self.model is None:
            raise RuntimeError("Model is not specified.")
        if hasattr(self.model, "fit"):
            if df.dropna().empty:
                raise RuntimeError("Valid values are not enough for training.")
            self.model.fit(df.dropna())

    def _predict_core(self, df):
        is_outliers = pd.Series(index=df.index)
        if not df.dropna().empty:
            if hasattr(self.model, "predict"):
                is_outliers.loc[df.dropna().index] = (
                    self.model.predict(df.dropna()) == -1
                )
            else:
                is_outliers.loc[df.dropna().index] = (
                    self.model.fit_predict(df.dropna()) == -1
                )
        return is_outliers


# =============================================================================
# PLEASE PUT PIPE-DERIVED DETECTOR CLASSES BELOW THIS LINE
# =============================================================================


class RegressionAD(_DetectorHD):
    """Detector that detects anomalous inter-series relationship.

    This detector performs regression to build relationship between a target
    series and the rest of series, and identifies a time point as anomalous
    when the residual of regression is beyond a threshold based on historical
    interquartile range.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    target: str, optional
        Name of the column to be regarded as target variable. If not specified,
        the first column in input DataFrame will be used.

    regressor: object
        Regressor to be used. Same as a scikit-learn regressor, it should
        minimally have `fit` and `predict` methods.

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

    """

    _default_params = {
        "target": None,
        "regressor": None,
        "c": 3.0,
        "side": "both",
    }

    def __init__(
        self,
        target=_default_params["target"],
        regressor=_default_params["regressor"],
        c=_default_params["c"],
        side=_default_params["side"],
    ):
        self.pipe_ = Pipenet(
            [
                {
                    "name": "regression_residual",
                    "model": RegressionResidual(
                        regressor=regressor, target=target
                    ),
                    "input": "original",
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
        super().__init__(regressor=regressor, target=target, side=side, c=c)
        self._sync_params()

    def _sync_params(self):
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps[0]["model"].regressor = self.regressor
        self.pipe_.steps[0]["model"].target = self.target
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

    def _predict_core(self, s):
        self._sync_params()
        return self.pipe_.detect(s)


class PcaAD(_DetectorHD):
    """Detector that detects outlier point with principal component analysis.

    This detector performs principal component analysis (PCA) to the
    multivariate time series (every time point is treated as a point in high-
    dimensional space), measures reconstruction error at every time point, and
    identifies a time point as anomalous when the recontruction error is beyond
    a threshold based on historical interquartile range.

    This detector is internally implemented as a `Pipeline` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    k: int, optional
        Number of principal components to use. Default: 1.

    c: float, optional
        Factor used to determine the bound of normal range based on historical
        interquartile range. Default: 5.0.

    Attributes
    ----------
    pipe_: adtk.pipe.Pipenet
        Internal pipenet object.
    """

    _default_params = {"k": 1, "c": 5.0}

    def __init__(self, k=_default_params["k"], c=_default_params["c"]):
        self.pipe_ = Pipeline(
            [
                ("pca_reconstruct_error", PcaReconstructionError(k=k)),
                ("ad", InterQuartileRangeAD(c=c)),
            ]
        )
        super().__init__(k=k, c=c)
        self._sync_params()

    def _sync_params(self):
        self.pipe_.steps[0][1].k = self.k
        self.pipe_.steps[1][1].c = self.c

    def _fit_core(self, s):
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s):
        self._sync_params()
        return self.pipe_.detect(s)
