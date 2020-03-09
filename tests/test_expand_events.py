import pandas as pd

from adtk.data import expand_events

events = [
    pd.Timestamp("2017-1-1 20:04:00"),
    (pd.Timestamp("2017-1-1 20:00:00"), pd.Timestamp("2017-1-1 20:05:59")),
    (pd.Timestamp("2017-1-1 20:03:00"), pd.Timestamp("2017-1-1 20:08:59")),
    pd.Timestamp("2017-1-1 20:30:00"),
    pd.Timestamp("2017-1-1 21:00:00"),
    (pd.Timestamp("2017-1-1 21:05:00"), pd.Timestamp("2017-1-1 21:06:59")),
    pd.Timestamp("2017-1-1 21:03:00"),
]


def test_expand_event_list():
    expanded_events = expand_events(
        events, left_expand="1min", right_expand="3min"
    )
    assert expanded_events == [
        (pd.Timestamp("2017-1-1 19:59:00"), pd.Timestamp("2017-1-1 20:11:59")),
        (pd.Timestamp("2017-1-1 20:29:00"), pd.Timestamp("2017-1-1 20:33:00")),
        (pd.Timestamp("2017-1-1 20:59:00"), pd.Timestamp("2017-1-1 21:09:59")),
    ]


def test_expand_event_dict():
    expanded_events = expand_events(
        {"A": events, "B": events}, left_expand="1min", right_expand="3min"
    )
    assert expanded_events == {
        "A": [
            (
                pd.Timestamp("2017-1-1 19:59:00"),
                pd.Timestamp("2017-1-1 20:11:59"),
            ),
            (
                pd.Timestamp("2017-1-1 20:29:00"),
                pd.Timestamp("2017-1-1 20:33:00"),
            ),
            (
                pd.Timestamp("2017-1-1 20:59:00"),
                pd.Timestamp("2017-1-1 21:09:59"),
            ),
        ],
        "B": [
            (
                pd.Timestamp("2017-1-1 19:59:00"),
                pd.Timestamp("2017-1-1 20:11:59"),
            ),
            (
                pd.Timestamp("2017-1-1 20:29:00"),
                pd.Timestamp("2017-1-1 20:33:00"),
            ),
            (
                pd.Timestamp("2017-1-1 20:59:00"),
                pd.Timestamp("2017-1-1 21:09:59"),
            ),
        ],
    }
