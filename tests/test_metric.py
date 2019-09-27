import pytest
import pandas as pd
from pandas import Timestamp
from adtk.metrics import recall, precision, iou

n = float("nan")

s_true = pd.Series(
    [0, 0, 1, 1, 0, 1, 0, n, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, n, 0, 0, 1, 0, 0],
    pd.date_range(start=0, periods=24, freq="1d"),
)
s_pred = pd.Series(
    [0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 0, 1, n, 1, 1, n, 0, 1, 0, 1, 1],
    pd.date_range(start=0, periods=24, freq="1d"),
)

df_true = pd.concat([s_true, s_pred], axis=1).rename(columns={0: "A", 1: "B"})
df_pred = pd.concat([s_pred, s_true], axis=1).rename(columns={0: "A", 1: "B"})


l_true = [
    (Timestamp("1970-01-03 00:00:00"), Timestamp("1970-01-04 00:00:00")),
    Timestamp("1970-01-06 00:00:00"),
    (Timestamp("1970-01-08 00:00:00"), Timestamp("1970-01-10 00:00:00")),
    Timestamp("1970-01-12 00:00:00"),
    (Timestamp("1970-01-14 00:00:00"), Timestamp("1970-01-18 00:00:00")),
    Timestamp("1970-01-22 00:00:00"),
]
l_pred = [
    (Timestamp("1970-01-02 00:00:00"), Timestamp("1970-01-07 00:00:00")),
    (Timestamp("1970-01-09 00:00:00"), Timestamp("1970-01-10 00:00:00")),
    Timestamp("1970-01-12 00:00:00"),
    Timestamp("1970-01-15 00:00:00"),
    (Timestamp("1970-01-17 00:00:00"), Timestamp("1970-01-19 00:00:00")),
    Timestamp("1970-01-21 00:00:00"),
    (Timestamp("1970-01-23 00:00:00"), Timestamp("1970-01-24 00:00:00")),
]

d_true = {"A": l_true, "B": l_pred}
d_pred = {"A": l_pred, "B": l_true}


def test_metric_series():
    assert recall(s_true, s_pred) == 9 / 12
    assert precision(s_true, s_pred) == 9 / 15
    assert iou(s_true, s_pred) == 9 / 17
    assert iou(s_pred, s_true) == 9 / 17


def test_metric_list():
    assert recall(l_true, l_pred) == 4 / 6
    assert precision(l_true, l_pred) == 4 / 7
    assert recall(l_true, l_pred, thresh=1) == 3 / 6
    assert precision(l_true, l_pred, thresh=1) == 3 / 7
    assert iou(l_true, l_pred) == 3 / 13


def test_metric_dataframe():
    assert recall(df_true, df_pred) == {"A": 9 / 12, "B": 9 / 15}
    assert precision(df_true, df_pred) == {"A": 9 / 15, "B": 9 / 12}
    assert iou(df_true, df_pred) == {"A": 9 / 17, "B": 9 / 17}


def test_metric_dict():
    assert recall(d_true, d_pred) == {"A": 4 / 6, "B": 4 / 7}
    assert precision(d_true, d_pred) == {"A": 4 / 7, "B": 4 / 6}
    assert recall(d_true, d_pred, thresh=1) == {"A": 3 / 6, "B": 3 / 7}
    assert precision(d_true, d_pred, thresh=1) == {"A": 3 / 7, "B": 3 / 6}
