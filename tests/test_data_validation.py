"""
Unit tests on data
"""
import numpy as np
import pandas as pd
import pytest

from adtk.data import validate_series

rand = np.random.RandomState(123)

regular_time_index = pd.date_range(start=0, periods=10, freq="1d")
so = pd.Series(np.arange(10), index=regular_time_index, name="value")
bo = pd.Series(
    [1, 0, 0, 0, 1, 1, 1, 0, 0, 0], index=regular_time_index, name="type1"
)
bom = pd.concat([bo, (1 - bo).rename("type2")], axis=1)
co = pd.Series(
    ["B", "A", "A", "A", np.nan, np.nan, np.nan, "B", "B", "B"],
    index=regular_time_index,
)
coi = pd.get_dummies(co)
con = pd.Series(
    ["B", "A", "A", "A", np.nan, np.nan, np.nan, "B", "B", "B"],
    index=regular_time_index,
    name="type3",
)
coni = pd.get_dummies(con, prefix="type3", prefix_sep="_")

test_targets = [
    (so, so),
    (bo, bo),
    (bom, bom),
    (co, coi),
    (con, coni),
    (pd.concat([so, bom, con], axis=1), pd.concat([so, bom, coni], axis=1)),
]


@pytest.mark.parametrize("x", test_targets)
def test_series_regular(x):
    # regular Series
    s = x[0].copy()
    sv = validate_series(s, check_categorical=True)
    if isinstance(sv, pd.Series):
        pd.testing.assert_series_equal(sv, x[1], check_dtype=False)
    elif isinstance(sv, pd.DataFrame):
        pd.testing.assert_frame_equal(sv, x[1], check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")
    # check if copy instead of view
    sc = s.copy()
    sv.iloc[0] == 1000
    if isinstance(s, pd.Series):
        pd.testing.assert_series_equal(s, sc, check_dtype=False)
    elif isinstance(s, pd.DataFrame):
        pd.testing.assert_frame_equal(s, sc, check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")


@pytest.mark.parametrize("x", test_targets)
def test_series_unsorted(x):
    # unsorted Series
    s = x[0].copy()
    s = s.iloc[[9, 6, 7, 1, 0, 3, 4, 5, 8, 2]]
    sv = validate_series(s, check_categorical=True)
    if isinstance(sv, pd.Series):
        pd.testing.assert_series_equal(sv, x[1], check_dtype=False)
    elif isinstance(sv, pd.DataFrame):
        pd.testing.assert_frame_equal(sv, x[1], check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")
    # check if copy instead of view
    sc = s.copy()
    sv.iloc[0] == 1000
    if isinstance(s, pd.Series):
        pd.testing.assert_series_equal(s, sc, check_dtype=False)
    elif isinstance(s, pd.DataFrame):
        pd.testing.assert_frame_equal(s, sc, check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")


@pytest.mark.parametrize("x", test_targets)
def test_series_duplicated_timestamp(x):
    # Series with duplicated time stamps
    s = x[0].copy()
    s = s.iloc[[0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7, 8, 9, 9]]
    sv = validate_series(s, check_categorical=True)
    if isinstance(sv, pd.Series):
        pd.testing.assert_series_equal(sv, x[1], check_dtype=False)
    elif isinstance(sv, pd.DataFrame):
        pd.testing.assert_frame_equal(sv, x[1], check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")
    # check if copy instead of view
    sc = s.copy()
    sv.iloc[0] == 1000
    if isinstance(s, pd.Series):
        pd.testing.assert_series_equal(s, sc, check_dtype=False)
    elif isinstance(s, pd.DataFrame):
        pd.testing.assert_frame_equal(s, sc, check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")


@pytest.mark.parametrize("x", test_targets)
def test_series_missed_timestamp(x):
    # Series with missed time stamps
    s = x[0].copy()
    s = s.iloc[[0, 1, 3, 4, 5, 6, 7, 9]]
    ss = x[1].copy()
    ss = ss.iloc[[0, 1, 3, 4, 5, 6, 7, 9]]
    sv = validate_series(s, check_categorical=True)
    if isinstance(sv, pd.Series):
        pd.testing.assert_series_equal(sv, ss, check_dtype=False)
    elif isinstance(sv, pd.DataFrame):
        pd.testing.assert_frame_equal(sv, ss, check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")
    # check if copy instead of view
    sc = s.copy()
    sv.iloc[0] == 1000
    if isinstance(s, pd.Series):
        pd.testing.assert_series_equal(s, sc, check_dtype=False)
    elif isinstance(s, pd.DataFrame):
        pd.testing.assert_frame_equal(s, sc, check_dtype=False)
    else:
        raise TypeError("Must be pandas Series or DataFrame")
