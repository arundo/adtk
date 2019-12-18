import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns


def plot(
    ts,
    anomaly_true,
    anomaly_pred,
    curve_group,
    anomaly_group,
    at_marker_one_curve,
):
    """
    ts
    anomaly_true: Series, list, DataFrame, dict of Series or list


    curve_group : "each", or "all", or [["A", "B"], ["B", "C"]], default "all"
    anomaly_group: "all", [["type i", "type ii"], ["type ii", "type iii"]], {"type i": "A", "type ii": ["A", "B"], "type iii": "C"}. default: "auto"
    at_marker_one_curve: None, bool
    """

    # TODO: ts as None

    # type check
    if isinstance(ts, pd.Series):
        if ts.name is None:
            df = ts.to_frame("Time Series")
        else:
            df = ts.to_frame()
    elif isinstance(ts, pd.DataFrame):
        df = ts.copy()
    else:
        raise TypeError("Argument `ts` must be a pandas Series or DataFrame.")

    # check duplicated column names
    if df.columns.duplicated().any():
        raise ValueError(
            "Input dataframe must not have duplicated column names."
        )

    # define subplots
    if curve_group == "each":
        curve_group = [[col] for col in df.columns]
    elif curve_group == "all":
        curve_group = [[col for col in df.columns]]
    else:
        curve_group = [
            ([group] if not isinstance(group, list) else group)
            for group in curve_group
        ]

    # prepare anomaly
    def _prepare_anomaly(anomaly):
        if anomaly is None:
            return None
        if isinstance(anomaly, (list, pd.Series):
            anomaly = {"Anomaly": anomaly}
        elif isinstance(anomaly, pd.DataFrame):
            anomaly = dict(anomaly)
        elif isintance(anomaly, dict):
            pass
        else:
            raise TypeError(
                "Anomaly must be a list, a dict, a pandas Series or DataFrame."
            )
        return anomaly
    anomaly_true = _prepare_anomaly(anomaly_true)
    anomaly_pred = _prepare_anomaly(anomaly_pred)

    if (anomaly_true is not None) and (anomaly_pred is not None):
        if anomaly_true.keys() ~= anomaly_pred.keys():
            raise ValueError("True and predicted anomaly are not consistent.")






