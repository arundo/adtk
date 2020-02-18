"""Check model fitting with short series
"""

import pytest
import numpy as np
import pandas as pd

from adtk.detector import (
    PersistAD,
    LevelShiftAD,
    VolatilityShiftAD,
    AutoregressionAD,
)

s = pd.Series(
    np.sin(np.arange(10)),
    index=pd.date_range(start="2017-1-1", periods=10, freq="D"),
)  # type: pd.Series


def test_persist_ad() -> None:
    model = PersistAD(window=10)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = PersistAD(window=9)
    model.fit(s)


def test_level_shift_ad() -> None:
    model = LevelShiftAD(window=6)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = PersistAD(window=5)
    model.fit(s)


def test_volatility_shift_ad() -> None:
    model = VolatilityShiftAD(window=6)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = PersistAD(window=5)
    model.fit(s)


def test_autoregression_ad() -> None:
    model = AutoregressionAD(n_steps=3, step_size=4)
    with pytest.raises(RuntimeError):
        model.fit(s)

    model = AutoregressionAD(n_steps=3, step_size=3)
    model.fit(s)
