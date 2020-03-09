"""
Unit tests on train-test split
"""
import numpy as np
import pandas as pd

from adtk.data import split_train_test


def test_split_series():
    """
    test all modes on a naive list of from 0 to 99
    """
    s = pd.Series(range(100))

    splits = split_train_test(s, mode=1, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
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

    splits = split_train_test(s, mode=2, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
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

    splits = split_train_test(s, mode=3, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
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

    splits = split_train_test(s, mode=4, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
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


def test_split_dataframe():
    """
    test all modes on a naive df of from 0 to 99
    """
    s = pd.Series(range(100))
    df = pd.DataFrame({"A": s, "B": s})

    splits = split_train_test(df, mode=1, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_train,
            [df.iloc[:20], df.iloc[25:45], df.iloc[50:70], df.iloc[75:95]],
        )
    )
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_test,
            [df.iloc[20:25], df.iloc[45:50], df.iloc[70:75], df.iloc[95:]],
        )
    )

    splits = split_train_test(df, mode=2, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_train, [df.iloc[:20], df.iloc[:40], df.iloc[:60], df.iloc[:80]]
        )
    )
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_test,
            [df.iloc[20:25], df.iloc[40:50], df.iloc[60:75], df.iloc[80:]],
        )
    )

    splits = split_train_test(df, mode=3, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_train, [df.iloc[:20], df.iloc[:40], df.iloc[:60], df.iloc[:80]]
        )
    )
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_test,
            [df.iloc[20:40], df.iloc[40:60], df.iloc[60:80], df.iloc[80:]],
        )
    )

    splits = split_train_test(df, mode=4, n_splits=4, train_ratio=0.8)
    ts_train, ts_test = zip(*splits)
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_train, [df.iloc[:20], df.iloc[:40], df.iloc[:60], df.iloc[:80]]
        )
    )
    assert all(
        np.array_equal(x.values, y.values)
        for x, y in zip(
            ts_test, [df.iloc[20:], df.iloc[40:], df.iloc[60:], df.iloc[80:]]
        )
    )
