from abc import ABC, abstractmethod
from copy import deepcopy

import pandas as pd


class _Model(ABC):
    _need_fit = True
    _default_params = {}

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
            self._fitted = False

    @abstractmethod
    def _fit(self, ts):
        pass

    @abstractmethod
    def _predict(self, ts):
        pass

    def get_params(self):
        """Get parameters of this model.

        Returns
        -------
        dict
            Model parameters.

        """
        return {key: getattr(self, key) for key in self._default_params.keys()}

    def set_params(self, **kwargs):
        """Set parameters of this model.

        Parameters
        ----------
        **kwargs
            Model parameters to set. If empty, then all parameters will be
            reset to default values.

        """
        for key, value in kwargs.items():
            if key not in self._default_params.keys():
                raise KeyError("{} is not a valid parameter".format(key))
        for key, value in kwargs.items():
            setattr(self, key, value)

        if not kwargs:
            for key, value in self._default_params.items():
                setattr(self, key, value)


class _Model1D(_Model):
    """Base class of _Detector1D and _Transformer1D."""

    def __init__(self, **kwargs):
        self._models = None
        super().__init__(**kwargs)

    def _update_models(self, cols):
        """Update attribute _models with given columns and model parameters.

        When a 1D model applied to a DataFrame, it is applied independently to
        each column of the DataFrame. Internally, a model object is created for
        each column. This private method is used to set those models.

        """
        # initialize or reset a model for each column
        if self._models is None:
            self._models = dict()
        for col in cols:
            if col not in self._models:
                self._models[col] = self.__class__()
            else:
                self._models[col].set_params()
        # remove models that correspond no column
        models_to_remove = list(
            filter(lambda key: key not in cols, self._models.keys())
        )
        for key in models_to_remove:
            self._models.pop(key)
        # for each parameter, set them over all models.
        for col in cols:
            self._models[col].set_params(**deepcopy(self.get_params()))

    def _fit(self, ts):
        if isinstance(ts, pd.Series):
            s = ts.copy()
            self._fit_core(s)
            self._models = None
        elif isinstance(ts, pd.DataFrame):
            df = ts.copy()
            if self._need_fit:
                self._update_models(df.columns)
                # fit model for each column
                for col in df.columns:
                    self._models[col].fit(df[col])
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")
        self._fitted = True

    def _predict(self, ts):
        if self._need_fit and (not self._fitted):
            raise RuntimeError("The model must be trained first.")
        if isinstance(ts, pd.Series):
            s = ts.copy()
            predicted = self._predict_core(s)
            # if a Series-to-Series operation, make sure Series name keeps
            if isinstance(predicted, pd.Series):
                predicted.name = ts.name
        elif isinstance(ts, pd.DataFrame):
            df = ts.copy()
            # if the model doesn't neef fit, initialize or reset a model for
            # each column
            if not self._need_fit:
                self._update_models(df.columns)
            # predict for each column
            predicted = pd.concat(
                [self._models[col]._predict(df[col]) for col in df.columns],
                axis=1,
            )
        else:
            raise TypeError("Input must be a pandas Series or DataFrame.")
        # make sure index freq is the same (because pandas has a bug that some
        # operation, e.g. concat, may change freq)
        predicted.index.freq = ts.index.freq
        return predicted

    @abstractmethod
    def _fit_core(self, ts):
        pass

    @abstractmethod
    def _predict_core(self, ts):
        pass

    @abstractmethod
    def fit(self, ts):
        pass

    @abstractmethod
    def predict(self, ts):
        pass

    @abstractmethod
    def fit_predict(self, ts):
        pass


class _ModelHD(_Model):
    def _fit(self, ts):
        if isinstance(ts, pd.DataFrame):
            df = ts.copy()
            self._fit_core(df)
        else:
            raise TypeError("Input must be a pandas DataFrame.")
        self._fitted = True

    def _predict(self, ts):
        if self._need_fit and (not self._fitted):
            raise RuntimeError("The model must be trained first.")
        if isinstance(ts, pd.DataFrame):
            df = ts.copy()
            predicted = self._predict_core(df)
        else:
            raise TypeError("Input must be a pandas DataFrame.")
        # make sure index freq is the same (because pandas has a bug that some
        # operation, e.g. concat, may change freq)
        predicted.index.freq = ts.index.freq
        return predicted

    @abstractmethod
    def _fit_core(self, ts):
        pass

    @abstractmethod
    def _predict_core(self, ts):
        pass

    @abstractmethod
    def fit(self, ts):
        pass

    @abstractmethod
    def predict(self, ts):
        pass

    @abstractmethod
    def fit_predict(self, ts):
        pass
