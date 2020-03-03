from typing import Union, List, Dict, Tuple
import pandas as pd
from ._base import _NonTrainableModel


class _Aggregator(_NonTrainableModel):
    def _predict(
        self,
        lists: Union[
            pd.DataFrame,
            Dict[str, Union[pd.Series, pd.DataFrame]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
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
        self,
        lists: Union[
            pd.DataFrame,
            Dict[str, Union[pd.Series, pd.DataFrame]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
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
            - If a dict of list, every value of the dict is a list of events
              where each event is represented as a pandas Timestamp if it is
              a singular time point or a 2-tuple of pandas Timestamps if it is
              a time interval.

        Returns
        -------
        list or a binary pandas Series
            Aggregated list of anomalies.

            - If input is a pandas DataFrame or a dict of Series/DataFrame,
              return a binary Series;
            - If input is a dict of list, return a list of events.

        """
        return self._predict(lists)

    def predict(
        self,
        lists: Union[
            pd.DataFrame,
            Dict[str, Union[pd.Series, pd.DataFrame]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
    ) -> Union[
        pd.Series, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]
    ]:
        """
        Alias of `aggregate`.
        """
        return self.aggregate(lists)
