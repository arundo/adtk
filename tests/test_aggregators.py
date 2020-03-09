import pandas as pd
from pandas import Timestamp

import adtk.aggregator as aggt


def test_or_dict_of_lists():
    """
    Test OrAggregator with input as a dict of lists of time stamps or time
    stamp 2-tuples
    """
    lists = {
        "A": [
            (Timestamp("2017-1-1"), Timestamp("2017-1-2")),
            (Timestamp("2017-1-5"), Timestamp("2017-1-8")),
            Timestamp("2017-1-10"),
        ],
        "B": [
            Timestamp("2017-1-2"),
            (Timestamp("2017-1-3"), Timestamp("2017-1-6")),
            Timestamp("2017-1-8"),
            (Timestamp("2017-1-7"), Timestamp("2017-1-9")),
            (Timestamp("2017-1-11"), Timestamp("2017-1-11")),
        ],
    }
    assert aggt.OrAggregator().aggregate(lists) == [
        (Timestamp("2017-01-01 00:00:00"), Timestamp("2017-01-02 00:00:00")),
        (Timestamp("2017-01-03 00:00:00"), Timestamp("2017-01-09 00:00:00")),
        Timestamp("2017-1-10"),
        Timestamp("2017-1-11"),
    ]

    lists = {
        "A": [
            (Timestamp("2017-1-1"), Timestamp("2017-1-2")),
            (Timestamp("2017-1-5"), Timestamp("2017-1-8")),
            Timestamp("2017-1-10"),
        ],
        "B": [],
    }
    assert aggt.OrAggregator().aggregate(lists) == [
        (Timestamp("2017-1-1"), Timestamp("2017-1-2")),
        (Timestamp("2017-1-5"), Timestamp("2017-1-8")),
        Timestamp("2017-1-10"),
    ]


def test_or_df():
    """
    Test OrAggregator with input as a DataFrame
    """
    df = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [0, 0], [float("nan"), 1], [0, float("nan")]],
        index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
    )
    pd.testing.assert_series_equal(
        aggt.OrAggregator().aggregate(df),
        pd.Series(
            [1, 1, 1, 0, 1, float("nan")],
            index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
        ),
    )


def test_or_dict_of_dfs():
    """
    Test OrAggregator with input as a dict of DataFrame
    """
    df1 = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [0, 0], [float("nan"), 1], [0, float("nan")]],
        index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
    )
    df2 = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [0, 0], [float("nan"), 1], [0, float("nan")]],
        index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
    )
    pd.testing.assert_series_equal(
        aggt.OrAggregator().aggregate({"A": df1, "B": df2}),
        pd.Series(
            [1, 1, 1, 0, 1, float("nan")],
            index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
        ),
    )


def test_and_dict_of_lists():
    """
    Test AndAggregator with input as a dict of lists of time stamps or time
    stamp 2-tuples
    """
    lists = {
        "A": [
            (Timestamp("2017-1-1"), Timestamp("2017-1-2")),
            (Timestamp("2017-1-5"), Timestamp("2017-1-8")),
            Timestamp("2017-1-10"),
        ],
        "B": [
            Timestamp("2017-1-2"),
            (Timestamp("2017-1-3"), Timestamp("2017-1-6")),
            Timestamp("2017-1-8"),
            (Timestamp("2017-1-7"), Timestamp("2017-1-9")),
            (Timestamp("2017-1-11"), Timestamp("2017-1-11")),
        ],
    }
    assert aggt.AndAggregator().aggregate(lists) == [
        Timestamp("2017-1-2"),
        (Timestamp("2017-01-05 00:00:00"), Timestamp("2017-01-06 00:00:00")),
        (Timestamp("2017-1-7 00:00:00"), Timestamp("2017-1-8 00:00:00")),
    ]

    lists = {
        "A": [
            (Timestamp("2017-1-1"), Timestamp("2017-1-2")),
            (Timestamp("2017-1-5"), Timestamp("2017-1-8")),
            Timestamp("2017-1-10"),
        ],
        "B": [],
    }
    assert aggt.AndAggregator().aggregate(lists) == []


def test_and_df():
    """
    Test AndAggregator with input as a DataFrame
    """
    df = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [0, 0], [float("nan"), 1], [0, float("nan")]],
        index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
    )
    pd.testing.assert_series_equal(
        aggt.AndAggregator().aggregate(df),
        pd.Series(
            [1, 0, 0, 0, float("nan"), 0],
            index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
        ),
    )


def test_and_dict_of_dfs():
    """
    Test AndAggregator with input as a dict of DataFrame
    """
    df1 = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [0, 0], [float("nan"), 1], [0, float("nan")]],
        index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
    )
    df2 = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [0, 0], [float("nan"), 1], [0, float("nan")]],
        index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
    )
    pd.testing.assert_series_equal(
        aggt.AndAggregator().aggregate({"A": df1, "B": df2}),
        pd.Series(
            [1, 0, 0, 0, float("nan"), 0],
            index=pd.date_range(start="2017-1-1", periods=6, freq="D"),
        ),
    )


def test_customized_aggregator():
    """
    Test customized aggregate
    """

    def myAggFunc(df, agg="and"):
        if agg == "and":
            return df.all(axis=1)
        elif agg == "or":
            return df.any(axis=1)
        else:
            raise ValueError("`agg` must be either 'and' or 'or'.")

    model = aggt.CustomizedAggregator(myAggFunc)

    df = pd.DataFrame(
        [[1, 1], [1, 0], [0, 1], [0, 0]],
        index=pd.date_range(start="2017-1-1", periods=4, freq="D"),
    )

    pd.testing.assert_series_equal(
        model.aggregate(df),
        pd.Series([True, False, False, False], index=df.index),
    )

    model.aggregate_func_params = {"agg": "or"}
    pd.testing.assert_series_equal(
        model.aggregate(df),
        pd.Series([True, True, True, False], index=df.index),
    )
