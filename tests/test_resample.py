import pandas as pd
import numpy as np

from adtk.data import resample


def test_resample_given_new_dt() -> None:
    s = pd.Series(
        [0, 1, 2, 3.5, 6, 9],
        index=[
            pd.Timestamp("2017-1-1 00:00:00"),
            pd.Timestamp("2017-1-1 00:01:00"),
            pd.Timestamp("2017-1-1 00:02:00"),
            pd.Timestamp("2017-1-1 00:03:30"),
            pd.Timestamp("2017-1-1 00:06:00"),
            pd.Timestamp("2017-1-1 00:09:00"),
        ],
    )

    s_new = resample(s.rename("A"), pd.Timedelta("1min"))
    pd.testing.assert_series_equal(
        s_new,
        pd.Series(
            range(10),
            index=pd.date_range(
                start="2017-1-1 00:00:00", periods=10, freq="min"
            ),
        ).rename("A"),
        check_dtype=False,
    )

    df_new = resample(pd.DataFrame({"A": s, "B": s}), pd.Timedelta("1min"))
    pd.testing.assert_frame_equal(
        df_new,
        pd.DataFrame(
            {"A": range(10), "B": range(10)},
            index=pd.date_range(
                start="2017-1-1 00:00:00", periods=10, freq="min"
            ),
        ),
        check_dtype=False,
    )


def test_resample_not_given_new_dt() -> None:
    s = pd.Series(
        [0, 1, 2, 3.5, 6, 9],
        index=[
            pd.Timestamp("2017-1-1 00:00:00"),
            pd.Timestamp("2017-1-1 00:01:00"),
            pd.Timestamp("2017-1-1 00:02:00"),
            pd.Timestamp("2017-1-1 00:03:30"),
            pd.Timestamp("2017-1-1 00:06:00"),
            pd.Timestamp("2017-1-1 00:09:00"),
        ],
    )

    s_new = resample(s.rename("A"))
    pd.testing.assert_series_equal(
        s_new,
        pd.Series(
            np.arange(0, 9.5, 0.5),
            index=pd.date_range(
                start="2017-1-1 00:00:00", periods=19, freq="30s"
            ),
        ).rename("A"),
        check_dtype=False,
    )

    df_new = resample(pd.DataFrame({"A": s, "B": s}))
    pd.testing.assert_frame_equal(
        df_new,
        pd.DataFrame(
            {"A": np.arange(0, 9.5, 0.5), "B": np.arange(0, 9.5, 0.5)},
            index=pd.date_range(
                start="2017-1-1 00:00:00", periods=19, freq="30s"
            ),
        ),
        check_dtype=False,
    )
