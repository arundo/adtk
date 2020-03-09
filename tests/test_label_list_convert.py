"""Test converter between binary labels and event lists
"""

import numpy as np
import pandas as pd
from pandas import Timestamp

from adtk.data import to_events, to_labels


def test_binary_label_to_list_freq_as_period_merge_consecutive():
    binary_series = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).T,
        index=pd.date_range(start=0, periods=10, freq="1d"),
        columns=["type1", "type2", "type3", "type4"],
    )

    anomaly_list = to_events(
        binary_series, freq_as_period=True, merge_consecutive=True
    )

    anomaly_list_true = {
        "type1": [
            (
                Timestamp("1970-01-02 00:00:00"),
                Timestamp("1970-01-02 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-06 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-09 23:59:59.999999999"),
            ),
        ],
        "type2": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-01 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-10 23:59:59.999999999"),
            ),
        ],
        "type3": [],
        "type4": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-10 23:59:59.999999999"),
            )
        ],
    }

    assert anomaly_list == anomaly_list_true

    for i in range(4):
        assert (
            to_events(binary_series["type{}".format(i + 1)])
            == anomaly_list_true["type{}".format(i + 1)]
        )


def test_binary_label_to_list_freq_as_period_not_merge_consecutive():
    binary_series = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).T,
        index=pd.date_range(start=0, periods=10, freq="1d"),
        columns=["type1", "type2", "type3", "type4"],
    )

    anomaly_list = to_events(
        binary_series, freq_as_period=True, merge_consecutive=False
    )

    anomaly_list_true = {
        "type1": [
            (
                Timestamp("1970-01-02 00:00:00"),
                Timestamp("1970-01-02 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-05 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-06 00:00:00"),
                Timestamp("1970-01-06 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-08 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-09 00:00:00"),
                Timestamp("1970-01-09 23:59:59.999999999"),
            ),
        ],
        "type2": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-01 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-08 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-09 00:00:00"),
                Timestamp("1970-01-09 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-10 00:00:00"),
                Timestamp("1970-01-10 23:59:59.999999999"),
            ),
        ],
        "type3": [],
        "type4": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-01 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-02 00:00:00"),
                Timestamp("1970-01-02 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-03 00:00:00"),
                Timestamp("1970-01-03 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-04 00:00:00"),
                Timestamp("1970-01-04 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-05 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-06 00:00:00"),
                Timestamp("1970-01-06 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-07 00:00:00"),
                Timestamp("1970-01-07 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-08 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-09 00:00:00"),
                Timestamp("1970-01-09 23:59:59.999999999"),
            ),
            (
                Timestamp("1970-01-10 00:00:00"),
                Timestamp("1970-01-10 23:59:59.999999999"),
            ),
        ],
    }

    assert anomaly_list == anomaly_list_true

    for i in range(4):
        assert (
            to_events(
                binary_series["type{}".format(i + 1)],
                freq_as_period=True,
                merge_consecutive=False,
            )
            == anomaly_list_true["type{}".format(i + 1)]
        )


def test_binary_label_to_list_freq_not_as_period_merge_consecutive():
    binary_series = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).T,
        index=pd.date_range(start=0, periods=10, freq="1d"),
        columns=["type1", "type2", "type3", "type4"],
    )

    anomaly_list = to_events(
        binary_series, freq_as_period=False, merge_consecutive=True
    )

    anomaly_list_true = {
        "type1": [
            Timestamp("1970-01-02 00:00:00"),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-06 00:00:00"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-09 00:00:00"),
            ),
        ],
        "type2": [
            Timestamp("1970-01-01 00:00:00"),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-10 00:00:00"),
            ),
        ],
        "type3": [],
        "type4": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-10 00:00:00"),
            )
        ],
    }

    assert anomaly_list == anomaly_list_true

    for i in range(4):
        assert (
            to_events(
                binary_series["type{}".format(i + 1)],
                freq_as_period=False,
                merge_consecutive=True,
            )
            == anomaly_list_true["type{}".format(i + 1)]
        )


def test_binary_label_to_list_freq_not_as_period_not_merge_consecutive():
    binary_series = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            ]
        ).T,
        index=pd.date_range(start=0, periods=10, freq="1d"),
        columns=["type1", "type2", "type3", "type4"],
    )

    anomaly_list = to_events(binary_series, freq_as_period=False)

    anomaly_list_true = {
        "type1": [
            Timestamp("1970-01-02 00:00:00"),
            Timestamp("1970-01-05 00:00:00"),
            Timestamp("1970-01-06 00:00:00"),
            Timestamp("1970-01-08 00:00:00"),
            Timestamp("1970-01-09 00:00:00"),
        ],
        "type2": [
            Timestamp("1970-01-01 00:00:00"),
            Timestamp("1970-01-08 00:00:00"),
            Timestamp("1970-01-09 00:00:00"),
            Timestamp("1970-01-10 00:00:00"),
        ],
        "type3": [],
        "type4": [
            Timestamp("1970-01-01 00:00:00"),
            Timestamp("1970-01-02 00:00:00"),
            Timestamp("1970-01-03 00:00:00"),
            Timestamp("1970-01-04 00:00:00"),
            Timestamp("1970-01-05 00:00:00"),
            Timestamp("1970-01-06 00:00:00"),
            Timestamp("1970-01-07 00:00:00"),
            Timestamp("1970-01-08 00:00:00"),
            Timestamp("1970-01-09 00:00:00"),
            Timestamp("1970-01-10 00:00:00"),
        ],
    }

    assert anomaly_list == anomaly_list_true

    for i in range(4):
        assert (
            to_events(
                binary_series["type{}".format(i + 1)],
                freq_as_period=False,
                merge_consecutive=False,
            )
            == anomaly_list_true["type{}".format(i + 1)]
        )


def test_list_to_label_freq_as_period():
    anomaly_list = {
        "type1": [
            (
                Timestamp("1970-01-02 00:00:00"),
                Timestamp("1970-01-02 00:00:00"),
            ),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-06 00:00:00"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-09 00:00:00"),
            ),
        ],
        "type2": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-01 00:00:00"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-10 00:00:00"),
            ),
        ],
        "type3": [],
        "type4": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-10 00:00:00"),
            )
        ],
        "type5": [
            (
                Timestamp("1969-12-12 00:00:05"),
                Timestamp("1969-12-25 00:00:05"),
            ),
            (
                Timestamp("1969-12-27 00:00:05"),
                Timestamp("1970-01-01 00:00:05"),
            ),
            (
                Timestamp("1970-01-02 00:00:05"),
                Timestamp("1970-01-04 00:00:05"),
            ),
            (
                Timestamp("1970-01-07 00:00:05"),
                Timestamp("1970-01-15 00:00:05"),
            ),
        ],
    }

    labels_true = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0, 0, 1, 1, 1, 1],
            ]
        ).T,
        columns=["type{}".format(i + 1) for i in range(5)],
        index=pd.date_range(start=0, periods=10, freq="1d"),
    ).astype(bool)

    for i in range(5):
        labels = to_labels(
            anomaly_list["type{}".format(i + 1)],
            pd.date_range(start=0, periods=10, freq="1d"),
            freq_as_period=True,
        )

        pd.testing.assert_series_equal(
            labels, labels_true["type{}".format(i + 1)], check_names=False
        )

    labels = to_labels(
        anomaly_list,
        pd.date_range(start=0, periods=10, freq="1d"),
        freq_as_period=True,
    )

    pd.testing.assert_frame_equal(
        labels.sort_index(axis=1), labels_true.sort_index(axis=1)
    )


def test_list_to_label_freq_not_as_period():
    anomaly_list = {
        "type1": [
            Timestamp("1970-01-02 00:00:00"),
            (
                Timestamp("1970-01-05 00:00:00"),
                Timestamp("1970-01-06 00:00:00"),
            ),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-09 00:00:00"),
            ),
        ],
        "type2": [
            Timestamp("1970-01-01 00:00:00"),
            (
                Timestamp("1970-01-08 00:00:00"),
                Timestamp("1970-01-10 00:00:00"),
            ),
        ],
        "type3": [],
        "type4": [
            (
                Timestamp("1970-01-01 00:00:00"),
                Timestamp("1970-01-10 00:00:00"),
            )
        ],
        "type5": [
            (
                Timestamp("1969-12-12 00:00:05"),
                Timestamp("1969-12-25 00:00:05"),
            ),
            (
                Timestamp("1969-12-27 00:00:05"),
                Timestamp("1970-01-01 00:00:05"),
            ),
            (
                Timestamp("1970-01-02 00:00:05"),
                Timestamp("1970-01-04 00:00:05"),
            ),
            (
                Timestamp("1970-01-07 00:00:05"),
                Timestamp("1970-01-15 00:00:05"),
            ),
        ],
    }

    labels_true = pd.DataFrame(
        np.array(
            [
                [0, 1, 0, 0, 1, 1, 0, 1, 1, 0],
                [1, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                [1, 0, 1, 1, 0, 0, 0, 1, 1, 1],
            ]
        ).T,
        columns=["type{}".format(i + 1) for i in range(5)],
        index=pd.date_range(start=0, periods=10, freq="1d"),
    ).astype(bool)

    for i in range(5):
        labels = to_labels(
            anomaly_list["type{}".format(i + 1)],
            pd.date_range(start=0, periods=10, freq="1d"),
            freq_as_period=False,
        )

        pd.testing.assert_series_equal(
            labels, labels_true["type{}".format(i + 1)], check_names=False
        )

    labels = to_labels(
        anomaly_list,
        pd.date_range(start=0, periods=10, freq="1d"),
        freq_as_period=False,
    )

    pd.testing.assert_frame_equal(
        labels.sort_index(axis=1), labels_true.sort_index(axis=1)
    )


def test_nan():
    s = pd.Series(
        [1, 1, 0, 0, 0, np.nan, 1, 1, np.nan, np.nan, 0, 1],
        index=pd.date_range(start="2017-1-1", periods=12, freq="D"),
    )
    anomaly_list = to_events(s)
    anomaly_list_true = [
        (
            Timestamp("2017-01-01 00:00:00"),
            Timestamp("2017-01-02 23:59:59.999999999"),
        ),
        (
            Timestamp("2017-01-07 00:00:00"),
            Timestamp("2017-01-08 23:59:59.999999999"),
        ),
        (
            Timestamp("2017-01-12 00:00:00"),
            Timestamp("2017-01-12 23:59:59.999999999"),
        ),
    ]
    assert anomaly_list == anomaly_list_true
