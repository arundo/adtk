"""We tested stl seasonal decomposition here instead in the transformer tests,
because small tolerance of numerical errors are allowed.
"""
import pandas as pd
import numpy as np
from adtk.transformer import STLDecomposition


def test_stl_deseasonal():
    s = pd.Series(
        np.arange(200) % 7 + np.arange(200),
        index=pd.date_range(start="2017-1-1", periods=200, freq="D"),
    )
    model = STLDecomposition()

    assert model.fit_transform(s).abs().max() < 1e-12

    model.fit(s.iloc[:100])
    assert model.transform(s.iloc[150:]).abs().max() < 1e-12

    model.fit(s.iloc[150:])
    assert model.transform(s.iloc[:100]).abs().max() < 1e-12

    model = STLDecomposition(freq=7 * 3)

    assert model.fit_transform(s).abs().max() < 1e-12

    model.fit(s.iloc[:100])
    assert model.transform(s.iloc[150:]).abs().max() < 1e-12

    model.fit(s.iloc[150:])
    assert model.transform(s.iloc[:100]).abs().max() < 1e-12
