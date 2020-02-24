from ._base import _Model1D, _ModelHD

import pandas as pd

from typing import Dict, Union, List, Any


class _Transformer1D(_Model1D):
    def fit(self, ts: Union[pd.Series, pd.DataFrame]) -> None:
        """Train the transformer with given time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used to train the transformer.
            If a DataFrame with k columns, k univariate transformers will be
            trained independently.

        """
        self._fit(ts)

    def transform(
        self, ts: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Transform time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be transformed.
            If a DataFrame with k columns, k univariate transformers will be
            applied to them independently.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        return self._predict(ts)

    def fit_transform(
        self, ts: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Train the transformer, and tranform the time series used for
        training.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used for training and be transformed.
            If a DataFrame with k columns, k univariate transformers will be
            applied to them independently.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        self.fit(ts)
        return self.predict(ts)

    def predict(
        self, ts: Union[pd.Series, pd.DataFrame], *args: Any, **kwargs: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Alias of `transform`.
        """
        return self.transform(ts)

    def fit_predict(
        self, ts: Union[pd.Series, pd.DataFrame], *args: Any, **kwargs: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Alias of `fit_transform`.
        """
        return self.fit_transform(ts)


class _TransformerHD(_ModelHD):
    def fit(self, df: pd.DataFrame) -> None:
        """Train the transformer with given time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used to train the transformer.

        """
        self._fit(df)

    def transform(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
        """Transform time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be transformed.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        return self._predict(df)

    def fit_transform(
        self, df: pd.DataFrame
    ) -> Union[pd.Series, pd.DataFrame]:
        """Train the transformer, and tranform the time series used for
        training.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used for training and be transformed.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        self.fit(df)
        return self.predict(df)

    def predict(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Alias of `transform`.
        """
        return self.transform(df)

    def fit_predict(
        self, df: pd.DataFrame, *args: Any, **kwargs: Any
    ) -> Union[pd.Series, pd.DataFrame]:
        """
        Alias of `fit_transform`.
        """
        return self.fit_transform(df)
