"""Check model predicting with short series
"""

import numpy as np
import pandas as pd

from adtk.detector import (
    AutoregressionAD,
    LevelShiftAD,
    PersistAD,
    VolatilityShiftAD,
)

s = pd.Series(
    np.sin(np.arange(100)),
    index=pd.date_range(start="2017-1-1", periods=100, freq="D"),
)


def test_persist_ad():
    model = PersistAD(window=1)
    s_train = s.copy().iloc[:-10]
    model.fit(s_train)

    s_test = s.copy().iloc[-2:]
    s_test.iloc[-1] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test), pd.Series([np.nan, 1.0], index=s_test.index)
    )

    s_test = s.copy().iloc[-1:]
    s_test.iloc[-1] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test), pd.Series([np.nan], index=s_test.index)
    )

    model = PersistAD(window=5)
    s_train = s.copy().iloc[:-10]
    model.fit(s_train)

    s_test = s.copy().iloc[-5:]
    s_test.iloc[-1] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test), pd.Series([np.nan] * 5, index=s_test.index)
    )

    s_test = s.copy().iloc[-6:]
    s_test.iloc[-1] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test),
        pd.Series([np.nan] * 5 + [1.0], index=s_test.index),
    )


def test_level_shift_ad():
    model = LevelShiftAD(window=3)
    s_train = s.copy().iloc[:-10]
    model.fit(s_train)

    s_test = s.copy().iloc[-5:]
    s_test.iloc[-3:] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test), pd.Series([np.nan] * 5, index=s_test.index)
    )

    s_test = s.copy().iloc[-6:]
    s_test.iloc[-3:] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test),
        pd.Series([np.nan] * 3 + [1.0] + [np.nan] * 2, index=s_test.index),
    )


def test_volatility_shift_ad():
    model = VolatilityShiftAD(window=3)
    s_train = s.copy().iloc[:-10]
    model.fit(s_train)

    s_test = s.copy().iloc[-5:]
    s_test.iloc[-3:] *= 10
    pd.testing.assert_series_equal(
        model.predict(s_test), pd.Series([np.nan] * 5, index=s_test.index)
    )

    s_test = s.copy().iloc[-6:]
    s_test.iloc[-3:] *= 10
    pd.testing.assert_series_equal(
        model.predict(s_test),
        pd.Series([np.nan] * 3 + [1.0] + [np.nan] * 2, index=s_test.index),
    )


def test_autoregression_ad():
    model = AutoregressionAD(n_steps=3, step_size=7)
    s_train = s.copy().iloc[:-10]
    model.fit(s_train)

    s_test = s.copy().iloc[-21:]
    s_test.iloc[-1:] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test), pd.Series([np.nan] * 21, index=s_test.index)
    )

    s_test = s.copy().iloc[-22:]
    s_test.iloc[-1:] = 10
    pd.testing.assert_series_equal(
        model.predict(s_test),
        pd.Series([np.nan] * 21 + [1.0], index=s_test.index),
    )
