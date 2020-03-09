from typing import Union

import pandas as pd

from ._base import (
    _NonTrainableMultivariateModel,
    _NonTrainableUnivariateModel,
    _TrainableMultivariateModel,
    _TrainableUnivariateModel,
)


class _NonTrainableUnivariateTransformer(_NonTrainableUnivariateModel):
    def predict(
        self, ts: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Transform time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be transformed. If a DataFrame with k columns, it is
            treated as k independent univariate time series and the transformer
            will be applied to each univariate series independently.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        return self._predict(ts)

    transform = predict


class _TrainableUnivariateTransformer(_TrainableUnivariateModel):
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

    def predict(
        self, ts: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Transform time series.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be transformed. If a DataFrame with k columns, it is
            treated as k independent univariate time series.

            - If the transformer was trained with a Series, the transformer
              will be applied to each univariate series independently;
            - If the transformer was trained with a DataFrame, i.e. the
              transformer is essentially k transformers, those transformers
              will be applied to each univariate series respectively.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        return self._predict(ts)

    def fit_predict(
        self, ts: Union[pd.Series, pd.DataFrame]
    ) -> Union[pd.Series, pd.DataFrame]:
        """Train the transformer, and tranform the time series used for
        training.

        Parameters
        ----------
        ts: pandas.Series or pandas.DataFrame
            Time series to be used for training and be transformed.
            If a DataFrame with k columns, it is treated as k independent
            univariate time series, and k univariate transformers will be
            trained and applied to each series independently.

        Returns
        -------
        pandas.Series or pandas.DataFrame
            Transformed time series.

        """
        self.fit(ts)
        return self.predict(ts)

    transform = predict
    fit_transform = fit_predict


class _NonTrainableMultivariateTransformer(_NonTrainableMultivariateModel):
    def predict(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
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

    transform = predict


class _TrainableMultivariateTransformer(_TrainableMultivariateModel):
    def fit(self, df: pd.DataFrame) -> None:
        """Train the transformer with given time series.

        Parameters
        ----------
        df: pandas.DataFrame
            Time series to be used to train the transformer.

        """
        self._fit(df)

    def predict(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
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

    def fit_predict(self, df: pd.DataFrame) -> Union[pd.Series, pd.DataFrame]:
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

    transform = predict
    fit_transform = fit_predict
