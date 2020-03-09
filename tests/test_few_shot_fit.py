"""Check model fitting with short series
"""

import numpy as np
import pandas as pd
import pytest

from adtk.detector import (
    AutoregressionAD,
    LevelShiftAD,
    PersistAD,
    VolatilityShiftAD,
)

s = pd.Series(
    np.sin(np.arange(10)),
    index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
)


def test_persist_ad():
    model = PersistAD(window=10)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = PersistAD(window=9)
    model.fit(s)


def test_level_shift_ad():
    model = LevelShiftAD(window=6)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = PersistAD(window=5)
    model.fit(s)


def test_volatility_shift_ad():
    model = VolatilityShiftAD(window=6)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = PersistAD(window=5)
    model.fit(s)


def test_autoregression_ad():
    model = AutoregressionAD(n_steps=3, step_size=4)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = AutoregressionAD(n_steps=3, step_size=3)
    model.fit(s)
