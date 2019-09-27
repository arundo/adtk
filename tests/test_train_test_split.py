"""
Unit tests on train-test split
"""
import pandas as pd
from adtk.data import split_train_test


def test_run_one_case_one_model():
    """
    test all modes on a naive list of from 0 to 99
    """
    s = pd.Series(range(100))

    ts_train, ts_test = split_train_test(
        s, mode=1, n_splits=4, train_ratio=0.8
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_train,
            [s.iloc[:20], s.iloc[25:45], s.iloc[50:70], s.iloc[75:95]],
        )
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_test, [s.iloc[20:25], s.iloc[45:50], s.iloc[70:75], s.iloc[95:]]
        )
    )

    ts_train, ts_test = split_train_test(
        s, mode=2, n_splits=4, train_ratio=0.8
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_train, [s.iloc[:20], s.iloc[:40], s.iloc[:60], s.iloc[:80]]
        )
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_test, [s.iloc[20:25], s.iloc[40:50], s.iloc[60:75], s.iloc[80:]]
        )
    )

    ts_train, ts_test = split_train_test(
        s, mode=3, n_splits=4, train_ratio=0.8
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_train, [s.iloc[:20], s.iloc[:40], s.iloc[:60], s.iloc[:80]]
        )
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_test, [s.iloc[20:40], s.iloc[40:60], s.iloc[60:80], s.iloc[80:]]
        )
    )

    ts_train, ts_test = split_train_test(
        s, mode=4, n_splits=4, train_ratio=0.8
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_train, [s.iloc[:20], s.iloc[:40], s.iloc[:60], s.iloc[:80]]
        )
    )
    assert all(
        x.equals(y)
        for x, y in zip(
            ts_test, [s.iloc[20:], s.iloc[40:], s.iloc[60:], s.iloc[80:]]
        )
    )
