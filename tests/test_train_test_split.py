"""
Unit tests on train-test split
"""
import pandas as pd
import numpy as np
from adtk.data import split_train_test


def test_split_series() -> None:
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


def test_split_dataframe() -> None:
    """
    test all modes on a naive df of from 0 to 99
    """
    s = pd.Series(range(100))
    df = pd.DataFrame({"A": s, "B": s})

    ts_train, ts_test = split_train_test(
        df, mode=1, n_splits=4, train_ratio=0.8
    )
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

    ts_train, ts_test = split_train_test(
        df, mode=2, n_splits=4, train_ratio=0.8
    )
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

    ts_train, ts_test = split_train_test(
        df, mode=3, n_splits=4, train_ratio=0.8
    )
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

    ts_train, ts_test = split_train_test(
        df, mode=4, n_splits=4, train_ratio=0.8
    )
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


def test_split_train_test_names() -> None:
    """
    test output series names or column names
    """
    s = pd.Series(range(100))

    ts_train, ts_test = split_train_test(
        s, mode=1, n_splits=4, train_ratio=0.8
    )
    assert all(
        [
            s_train.name == "train_{}".format(i)
            for i, s_train in enumerate(ts_train)
        ]
    )
    assert all(
        [
            s_test.name == "test_{}".format(i)
            for i, s_test in enumerate(ts_test)
        ]
    )

    ts_train, ts_test = split_train_test(
        s.rename("C"), mode=1, n_splits=4, train_ratio=0.8
    )
    assert all(
        [
            s_train.name == "C_train_{}".format(i)
            for i, s_train in enumerate(ts_train)
        ]
    )
    assert all(
        [
            s_test.name == "C_test_{}".format(i)
            for i, s_test in enumerate(ts_test)
        ]
    )

    ts_train, ts_test = split_train_test(
        pd.DataFrame({"A": s, "B": s}), mode=1, n_splits=4, train_ratio=0.8
    )
    assert all(
        [
            all(
                df_train.columns
                == ["A_train_{}".format(i), "B_train_{}".format(i)]
            )
            for i, df_train in enumerate(ts_train)
        ]
    )
    assert all(
        [
            all(
                df_test.columns
                == ["A_test_{}".format(i), "B_test_{}".format(i)]
            )
            for i, df_test in enumerate(ts_test)
        ]
    )

