"""Module for metrics that scores predicted anomaly list against ground truth.
"""

import pandas as pd

from ..aggregator import AndAggregator, OrAggregator
from ..data import validate_events

__all__ = ["recall", "precision", "f1_score", "iou"]


def recall(y_true, y_pred, thresh=0.5):
    """Recall score of prediction.

    Recall, a.k.a. sensitivity, hit rate, or true positive rate (TPR), is the
    percentage of true anomalies that are detected successfully.

    When the input is anomaly labels, metric calculation treats every time
    point with positive label as an independent event. An anomalous time point
    that are detected as anomalous is considered as a successful detection.

    When the input is lists of anomalous time windows, metric calculation
    treats every anomalous time window as an independent event. A true event is
    considered as successfully detected if the percentage of this time window
    included in the detected list is greater or equal to `thresh`. Note that
    input time windows will be merged first if overlapped time windows exists
    in the list.

    Parameters
    ----------
    y_true: pandas Series or DataFrame, list, or dict
        Labels or lists of true anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    y_pred: pandas Series or DataFrame, list, or dict
        Labels or lists of predicted anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    thresh: float, optional
        Threshold of a hit. Only used if input is list or dict. Default: 0.5.

    Returns
    -------
    float or dict
        Score(s) for each type of anomaly.

    """
    if (thresh <= 0) or (thresh > 1):
        raise ValueError(
            "Parameter `thresh` must be a positive number less or equal to 1."
        )
    if type(y_true) != type(y_pred):
        raise TypeError("y_true and y_pred must have same type.")

    if isinstance(y_true, pd.Series):
        try:
            pd.testing.assert_index_equal(y_true.index, y_pred.index)
        except AssertionError:
            raise ValueError("y_true and y_pred must have identical index")
        if y_true.clip(0, 1).round().sum() != 0:
            return (
                y_true.clip(0, 1).round() * y_pred.clip(0, 1).round()
            ).sum() / (y_true.clip(0, 1).round()).sum()
        else:
            return float("nan")
    elif isinstance(y_true, pd.DataFrame):
        if set(y_true.columns) != set(y_pred.columns):
            raise ValueError("y_true and y_pred must have identical columns.")
        return {
            col: recall(y_true[col], y_pred[col]) for col in y_true.columns
        }
    elif isinstance(y_true, list):
        y_true = validate_events(y_true)
        if not y_true:
            return float("nan")
        y_int = AndAggregator().aggregate({"y_true": y_true, "y_pred": y_pred})
        n_hit = 0
        for w_true in y_true:
            if isinstance(w_true, tuple):
                true_start = w_true[0]
                true_end = w_true[1]
            else:
                true_start = w_true
                true_end = w_true
            len_w_true = true_end - true_start
            if len_w_true > pd.Timedelta(0):
                len_overlap = pd.Timedelta(0)
                for w_int in y_int:
                    if isinstance(w_int, tuple):
                        int_start = w_int[0]
                        int_end = w_int[1]
                    else:
                        int_start = w_int
                        int_end = w_int
                    len_overlap += max(
                        pd.Timedelta(0),
                        min(true_end, int_end) - max(true_start, int_start),
                    )
                if len_overlap >= thresh * len_w_true:
                    n_hit += 1
            else:
                for w_int in y_int:
                    if isinstance(w_int, tuple):
                        int_start = w_int[0]
                        int_end = w_int[1]
                    else:
                        int_start = w_int
                        int_end = w_int
                    if (int_start <= true_start) and (int_end >= true_end):
                        n_hit += 1
                        break
        return n_hit / len(y_true)
    elif isinstance(y_true, dict):
        return {
            key: recall(y_true[key], y_pred[key], thresh=thresh)
            for key in y_true.keys()
        }
    else:
        raise TypeError(
            "y_true and y_pred must be pandas Series or DataFrame, list, or a "
            "dict of lists"
        )


def precision(y_true, y_pred, thresh=0.5):
    """Precision score of prediction.

    Precision, a.k.a. positive predictive value (PPV), is the percentage of
    predicted anomalies that are true anomalies.

    When the input is anomaly labels, metric calculation treats every time
    point with positive label as an independent event. An anomalous time point
    that are detected as anomalous is considered as a successful detection.

    When the input is lists of anomalous time windows, metric calculation
    treats every anomalous time window as an independent event. A detected
    event is considered as a successfully detection if the percentage of this
    time window included in the true anomaly list is greater or equal to
    `thresh`. Note that input time windows will be merged first if overlapped
    time windows exists in the list.

    Parameters
    ----------
    y_true: pandas Series or DataFrame, list, or dict
        Labels or lists of true anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    y_pred: pandas Series or DataFrame, list, or dict
        Labels or lists of predicted anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    thresh: float, optional
        Threshold of a hit. Only used if input is list or dict. Default: 0.5.

    Returns
    -------
    float or dict
        Score(s) for each type of anomaly.

    """
    return recall(y_pred, y_true, thresh=thresh)


def f1_score(y_true, y_pred, recall_thresh=0.5, precision_thresh=0.5):
    """F1 score of prediction.

    F1 score is the harmonic mean of precision and recall. For more details
    about precision and recall, please refer to function `precision` and
    `recall`.

    Parameters
    ----------
    y_true: pandas Series or DataFrame, list, or dict
        Labels or lists of true anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    y_pred: pandas Series or DataFrame, list, or dict
        Labels or lists of predicted anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    recall_thresh: float, optional
        Threshold of recall calculation. Only used if input is list or dict.
        For more details, please refer to function `recall`. Default: 0.5.

    precision_thresh: float, optional
        Threshold of precision calculation. Only used if input is list or dict.
        For more details, please refer to function `precision`. Default: 0.5.

    Returns
    -------
    float or dict
        Score(s) for each type of anomaly.

    """
    recall_score = recall(y_true, y_pred, recall_thresh)
    precision_score = precision(y_true, y_pred, precision_thresh)
    if recall_score + precision_score != 0:
        return (
            2
            * recall_score
            * precision_score
            / (recall_score + precision_score)
        )
    else:
        return float("nan")


def iou(y_true, y_pred):
    """IoU (Intersection over Union) score of prediction.

    Intersection over union is the length ratio between time segments that are
    identified as anomalous in both lists and those identified by at least one
    of the two lists.

    When the input is anomaly labels, metric calculation counts the number of
    anomalous time points.

    When the input is lists of anomalous time windows, metric calculation
    measure the length of time segments.

    Parameters
    ----------
    y_true: pandas Series or DataFrame, list, or dict
        Labels or lists of true anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    y_pred: pandas Series or DataFrame, list, or dict
        Labels or lists of predicted anomalies.
        If pandas Series, it is treated as binary labels along time index.
        If pandas DataFrame, each column is treated as a type of anomaly.
        If list, it is treated as a list of anomalous time windows.
        If dict of lists, each item is treated as a type of anomaly.

    Returns
    -------
    float or dict
        Score(s) for each type of anomaly.

    """
    if type(y_true) != type(y_pred):
        raise TypeError("y_true and y_pred must have same type.")

    if isinstance(y_true, pd.Series):
        try:
            pd.testing.assert_index_equal(y_true.index, y_pred.index)
        except AssertionError:
            raise ValueError("y_true and y_pred must have identical index")
        if (y_true.clip(0, 1).round() + y_pred.clip(0, 1).round()).clip(
            0, 1
        ).sum() != 0:
            return (
                (y_true.clip(0, 1).round() * y_pred.clip(0, 1).round()).sum()
                / (y_true.clip(0, 1).round() + y_pred.clip(0, 1).round())
                .clip(0, 1)
                .sum()
            )
        else:
            return float("nan")
    elif isinstance(y_true, pd.DataFrame):
        if set(y_true.columns) != set(y_pred.columns):
            raise ValueError("y_true and y_pred must have identical columns.")
        return {col: iou(y_true[col], y_pred[col]) for col in y_true.columns}
    elif isinstance(y_true, list):
        y_int = AndAggregator().aggregate({"y_true": y_true, "y_pred": y_pred})
        y_union = OrAggregator().aggregate(
            {"y_true": y_true, "y_pred": y_pred}
        )
        len_int = sum(
            [
                (w[1] - w[0]).total_seconds() if isinstance(w, tuple) else 0
                for w in y_int
            ]
        )
        len_union = sum(
            [
                (w[1] - w[0]).total_seconds() if isinstance(w, tuple) else 0
                for w in y_union
            ]
        )
        if len_union == 0:
            return float("nan")
        return len_int / len_union
    elif isinstance(y_true, dict):
        return {key: iou(y_true[key], y_pred[key]) for key in y_true.keys()}
    else:
        raise TypeError(
            "y_true and y_pred must be pandas Series or DataFrame, list, or a "
            "dict of lists"
        )
