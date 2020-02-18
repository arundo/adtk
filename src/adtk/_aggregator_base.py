import pandas as pd

from ._base import _Model

from typing import Any, Union, List, Dict


class _Aggregator(_Model):
    _need_fit = False  # type: bool

    def _fit(self, lists: Union[pd.DataFrame, Dict[Any, Any]]) -> None:
        pass

    def _predict(
        self, lists: Union[pd.DataFrame, Dict[Any, Any]]
    ) -> Union[pd.DataFrame, pd.Series]:
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

    def aggregate(
        self, lists: Union[pd.DataFrame, Dict[Any, Any]]
    ) -> Union[List[pd.Timestamp], pd.Series]:
        """Aggregate multiple lists of anomalies into one.

        Parameters
        ----------
        lists: pandas.DataFrame, a dict of Series/DataFrame, or a dict of lists
            Anomaly lists to be aggregated.

            - If a pandas DataFrame, every column is a binary Series
              representing a list of anomalies;
            - If a dict of Series/DataFrame, every value of the dict is a
              binary Series/DataFrame representing a list / a set of lists of
              anomalies;
            - If a dict of list, every value of the dict is a list of pandas
              Timestamps representing anomalous time points.

        Returns
        -------
        list of pandas TimeStamps, or a binary pandas Series
            Aggregated list of anomalies.

            - If input is a pandas DataFrame or a dict of Series/DataFrame,
              return a binary Series;
            - If input is a dict of list, return a list.

        """
        return self._predict(lists)

    def predict(
        self,
        lists: Union[pd.DataFrame, Dict[Any, Any]],
        *args: Any,
        **kwargs: Any
    ) -> Union[List[pd.Timestamp], pd.Series]:
        """
        Alias of `aggregate`.
        """
        return self.aggregate(lists)

    def fit_predict(
        self,
        lists: Union[pd.DataFrame, Dict[Any, Any]],
        *args: Any,
        **kwargs: Any
    ) -> Union[List[pd.Timestamp], pd.Series]:
        """
        Alias of `aggregate`.
        """
        return self.aggregate(lists)
