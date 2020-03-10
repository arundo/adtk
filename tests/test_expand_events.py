import pandas as pd

from adtk.data import expand_events

event_list = [
    pd.Timestamp("2017-1-1 20:04:00"),
    (pd.Timestamp("2017-1-1 20:00:00"), pd.Timestamp("2017-1-1 20:05:59")),
    (pd.Timestamp("2017-1-1 20:03:00"), pd.Timestamp("2017-1-1 20:08:59")),
    pd.Timestamp("2017-1-1 20:30:00"),
    pd.Timestamp("2017-1-1 21:00:00"),
    (pd.Timestamp("2017-1-1 21:05:00"), pd.Timestamp("2017-1-1 21:06:59")),
    pd.Timestamp("2017-1-1 21:03:00"),
]

nan = float("nan")
event_labels = pd.Series(
    [0, 0, 1, 1, nan, 0, 1, 0, nan, 0, 0, 1],
    index=pd.date_range(start="2017-1-1", periods=12, freq="D"),
)


def test_expand_event_series_freq():
    expanded_events = expand_events(
        event_labels,
        left_expand="1hour",
        right_expand="1hour",
        freq_as_period=True,
    )
    true_expanded_events = pd.Series(
        [0, 1, 1, 1, 1, 1, 1, 1, nan, 0, 1, 1],
        index=pd.date_range(start="2017-1-1", periods=12, freq="D"),
    )
    pd.testing.assert_series_equal(
        true_expanded_events, expanded_events, check_dtype=False
    )


def test_expand_event_series_no_freq():
    expanded_events = expand_events(
        event_labels,
        left_expand="1hour",
        right_expand="1hour",
        freq_as_period=False,
    )
    pd.testing.assert_series_equal(
        event_labels, expanded_events, check_dtype=False
    )


def test_expand_event_df_freq():
    expanded_events = expand_events(
        pd.concat(
            [event_labels.rename("A"), event_labels.rename("B")], axis=1
        ),
        left_expand="1hour",
        right_expand="1hour",
        freq_as_period=True,
    )
    true_expanded_events = pd.Series(
        [0, 1, 1, 1, 1, 1, 1, 1, nan, 0, 1, 1],
        index=pd.date_range(start="2017-1-1", periods=12, freq="D"),
    )
    true_expanded_events = pd.concat(
        [true_expanded_events.rename("A"), true_expanded_events.rename("B")],
        axis=1,
    )
    pd.testing.assert_frame_equal(
        true_expanded_events, expanded_events, check_dtype=False
    )


def test_expand_event_df_no_freq():
    expanded_events = expand_events(
        pd.concat(
            [event_labels.rename("A"), event_labels.rename("B")], axis=1
        ),
        left_expand="1hour",
        right_expand="1hour",
        freq_as_period=False,
    )

    pd.testing.assert_frame_equal(
        pd.concat(
            [event_labels.rename("A"), event_labels.rename("B")], axis=1
        ),
        expanded_events,
        check_dtype=False,
    )


def test_expand_event_list():
    expanded_events = expand_events(
        event_list, left_expand="1min", right_expand="3min"
    )
    assert expanded_events == [
        (pd.Timestamp("2017-1-1 19:59:00"), pd.Timestamp("2017-1-1 20:11:59")),
        (pd.Timestamp("2017-1-1 20:29:00"), pd.Timestamp("2017-1-1 20:33:00")),
        (pd.Timestamp("2017-1-1 20:59:00"), pd.Timestamp("2017-1-1 21:09:59")),
    ]


def test_expand_event_dict():
    expanded_events = expand_events(
        {"A": event_list, "B": event_list},
        left_expand="1min",
        right_expand="3min",
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
