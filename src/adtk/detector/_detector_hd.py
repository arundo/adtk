"""Module for high-dimensional detectors.

High-dimensional detectors detect anomalies from high-dimensional time series,
i.e. from pandas DataFrame.
"""

from collections import Counter
from typing import Any, Callable, Dict, Optional, Tuple

import pandas as pd

from .._detector_base import _TrainableMultivariateDetector
from ..aggregator import AndAggregator
from ..detector import InterQuartileRangeAD, ThresholdAD
from ..pipe import Pipeline, Pipenet
from ..transformer import (
    CustomizedTransformer1D,
    PcaReconstructionError,
    RegressionResidual,
)


class CustomizedDetectorHD(_TrainableMultivariateDetector):
    """Multivariate detector derived from a user-given function and parameters.

    Parameters
    ----------
    detect_func: function
        A function detecting anomalies from multivariate time series.

        The first input argument must be a pandas DataFrame, optional input
        argument may be accepted through parameter `detect_func_params` and the
        output of `fit_func`, and the output must be a binary pandas Series
        with the same index as input.

    detect_func_params: dict, optional
        Parameters of `detect_func`. Default: None.

    fit_func: function, optional
        A function training parameters of `detect_func` with multivariate time
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
        self._fitted_detect_func_params = {}  # type: Dict
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

    def _fit_core(self, df: pd.DataFrame) -> None:
        if self.fit_func is not None:
            if self.fit_func_params is not None:
                fit_func_params = self.fit_func_params
            else:
                fit_func_params = {}
            self._fitted_detect_func_params = self.fit_func(
                df, **fit_func_params
            )

    def _predict_core(self, df: pd.DataFrame) -> pd.Series:
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


class MinClusterDetector(_TrainableMultivariateDetector):
    """Detector that detects anomaly based on clustering of historical data.

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

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("model",)

    def _fit_core(self, df: pd.DataFrame) -> None:
        if df.dropna().empty:
            raise RuntimeError("Valid values are not enough for training.")
        clustering_result = self.model.fit_predict(df.dropna())
        cluster_count = Counter(clustering_result)  # type: Counter
        self._anomalous_cluster_id = cluster_count.most_common()[-1][0]

    def _predict_core(self, df: pd.DataFrame) -> pd.Series:
        cluster_id = pd.Series(float("nan"), index=df.index)
        if not df.dropna().empty:
            cluster_id.loc[df.dropna().index] = self.model.predict(df.dropna())
        predicted = pd.Series(
            cluster_id == self._anomalous_cluster_id, index=df.index
        )
        predicted[cluster_id.isna()] = float("nan")
        return predicted


class OutlierDetector(_TrainableMultivariateDetector):
    """Detector that detects anomaly based on a outlier detection model.

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

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("model",)

    def _fit_core(self, df: pd.DataFrame) -> None:
        if hasattr(self.model, "fit"):
            if df.dropna().empty:
                raise RuntimeError("Valid values are not enough for training.")
            self.model.fit(df.dropna())

    def _predict_core(self, df: pd.DataFrame) -> pd.Series:
        is_outliers = pd.Series(float("nan"), index=df.index)
        if not df.dropna().empty:
            if hasattr(self.model, "predict"):
                is_outliers.loc[df.dropna().index] = (
                    self.model.predict(df.dropna()) == -1
                )
            else:
                is_outliers.loc[df.dropna().index] = (
                    self.model.fit_predict(df.dropna()) == -1
                )
        predicted = pd.Series(is_outliers == 1, index=df.index)
        predicted[is_outliers.isna()] = float("nan")
        return predicted


# =============================================================================
# PLEASE PUT PIPE-DERIVED DETECTOR CLASSES BELOW THIS LINE
# =============================================================================


class RegressionAD(_TrainableMultivariateDetector):
    """Detector that detects anomalous inter-series relationship.

    This detector performs regression to build relationship between a target
    series and the rest of series, and identifies a time point as anomalous
    when the residual of regression is anomalously large.

    This detector is internally implemented as a `Pipenet` object. Advanced
    users may learn more details by checking attribute `pipe_`.

    Parameters
    ----------
    target: str
        Name of the column to be regarded as target variable.

    regressor: object
        Regressor to be used. Same as a scikit-learn regressor, it should
        minimally have `fit` and `predict` methods.

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
        self, regressor: Any, target: str, c: float = 3.0, side: str = "both"
    ) -> None:
        self.pipe_ = Pipenet(
            {
                "regression_residual": {
                    "model": RegressionResidual(
                        regressor=regressor, target=target
                    ),
                    "input": "original",
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
        self.regressor = regressor
        self.target = target
        self.side = side
        self.c = c
        self._sync_params()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("regressor", "target", "c", "side")

    def _sync_params(self) -> None:
        if self.side not in ["both", "positive", "negative"]:
            raise ValueError(
                "Parameter `side` must be 'both', 'positive' or 'negative'."
            )
        self.pipe_.steps["regression_residual"][
            "model"
        ].regressor = self.regressor
        self.pipe_.steps["regression_residual"]["model"].set_params(
            target=self.target
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

    def _fit_core(self, s: pd.DataFrame) -> None:
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s: pd.DataFrame) -> pd.Series:
        self._sync_params()
        return self.pipe_.detect(s)


class PcaAD(_TrainableMultivariateDetector):
    """Detector that detects outlier point with principal component analysis.

    This detector performs principal component analysis (PCA) to the
    multivariate time series (every time point is treated as a point in high-
    dimensional space), measures reconstruction error at every time point, and
    identifies a time point as anomalous when the recontruction error is beyond
    anomalously large.

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

    def __init__(self, k: int = 1, c: float = 5.0) -> None:
        self.pipe_ = Pipeline(
            [
                ("pca_reconstruct_error", PcaReconstructionError(k=k)),
                ("ad", InterQuartileRangeAD(c=c)),
            ]
        )
        super().__init__()
        self.k = k
        self.c = c
        self._sync_params()

    @property
    def _param_names(self) -> Tuple[str, ...]:
        return ("k", "c")

    def _sync_params(self) -> None:
        self.pipe_.steps[0][1].set_params(k=self.k)
        self.pipe_.steps[1][1].set_params(c=self.c)

    def _fit_core(self, s: pd.DataFrame) -> None:
        self._sync_params()
        self.pipe_.fit(s)

    def _predict_core(self, s: pd.DataFrame) -> pd.Series:
        self._sync_params()
        return self.pipe_.detect(s)
