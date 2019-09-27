import pandas as pd

from ._base import _Model


class _Aggregator(_Model):
    _need_fit = False

    def _fit(self, lists):
        pass

    def _predict(self, lists):
        if isinstance(lists, dict):
            if not (
                all([isinstance(lst, list) for lst in lists.values()])
                or all(
                    [
                        isinstance(lst, (pd.Series, pd.DataFrame))
                        for lst in lists.values()
                    ]
                )
            ):
                raise TypeError(
                    "Input must be a pandas DataFrame, a dict of lists, or a "
                    "dict of pandas Series/DataFrame."
                )
        elif isinstance(lists, pd.DataFrame):
            pass
        else:
            raise TypeError(
                "Input must be a pandas DataFrame, a dict of lists, or a dict "
                "of pandas Series/DataFrame."
            )
        return self._predict_core(lists)

    def aggregate(self, lists):
        """Aggregate mulitple lists of anomalies into one.

        Parameters
        ----------
        lists: pandas.DataFrame, a dict of Series/DataFrame, or a dict of lists
            Anomaly lists to be aggregated.
            If a pandas DataFrame, every column is a binary Series representing
            a list of anomalies;
            If a dict of Series/DataFrame, every value of the dict is a binary
            Series/DataFrame representing a list / a set of lists of anomalies;
            If a dict of list, every value of the dict is a list of pandas
            Timestamps representing anomalous time points.

        Returns
        -------
        list of pandas TimeStamps, or a binary pandas Series
            Aggregated list of anomalies.
            If input is a pandas DataFrame or a dict of Series/DataFrame,
            return a binary Series;
            If input is a dict of list, return a list.

        """
        return self._predict(lists)

    def predict(self, lists, *args, **kwargs):
        """
        Alias of `aggregate`.
        """
        return self.aggregate(lists)

    def fit_predict(self, lists, *args, **kwargs):
        """
        Alias of `aggregate`.
        """
        return self.aggregate(lists)
