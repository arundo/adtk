from typing import Dict, List, Tuple, Union

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
        """Aggregate multiple lists of anomalies into one.

        Parameters
        ----------
        lists: pandas.DataFrame, a dict of Series/DataFrame, or a dict of lists
            Anomaly lists to be aggregated.

            - If a pandas DataFrame, every column is a binary Series
              representing a type of anomaly.
            - If a dict of pandas Series/DataFrame, every value of the dict is
              a binary Series/DataFrame representing a type or some types of
              anomaly;
            - If a dict of list, every value of the dict is a type of anomaly
              as a list of events, where each event is represented as a pandas
              Timestamp if it is instantaneous or a 2-tuple of pandas
              Timestamps if it is a closed time interval.

        Returns
        -------
        list or a binary pandas Series
            Aggregated list of anomalies.

            - If input is a pandas DataFrame or a dict of Series/DataFrame,
              return a single binary pandas Series;
            - If input is a dict of lists, return a single list of events.

        """
        return self._predict(lists)

    aggregate = predict
