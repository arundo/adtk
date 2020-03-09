"""
Test raising error when training and testing dataframes are inconsistent in
multivariate trainable models.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import LocalOutlierFactor

import adtk.detector as detector
import adtk.transformer as transformer

models = [
    detector.MinClusterDetector(KMeans(n_clusters=2)),
    detector.OutlierDetector(
        LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    ),
    detector.RegressionAD(target="A", regressor=LinearRegression()),
    detector.PcaAD(),
    transformer.RegressionResidual(target="A", regressor=LinearRegression()),
    transformer.PcaReconstructionError(),
    transformer.PcaProjection(),
    transformer.PcaReconstruction(),
]

df_train = pd.DataFrame(
    np.arange(40).reshape(20, 2),
    columns=["A", "B"],
    index=pd.date_range(start="2017-1-1", periods=20, freq="D"),
)

df_test_ok = pd.DataFrame(
    np.arange(0, -60, -1).reshape(20, 3),
    columns=["C", "B", "A"],
    index=pd.date_range(start="2017-1-1", periods=20, freq="D"),
)

df_test_not_ok = pd.DataFrame(
    np.arange(0, -60, -1).reshape(20, 3),
    columns=["C", "D", "A"],
    index=pd.date_range(start="2017-1-1", periods=20, freq="D"),
)


@pytest.mark.parametrize("model", models)
def test_inconsistent_train_test(model):
    model.fit(df_train)

    model.predict(df_test_ok)

    with pytest.raises(
        ValueError,
        match="The model was trained by a pandas DataFrame with columns",
    ):
        model.predict(df_test_not_ok)
