"""Module for model pipeline and pipenet.

Pipeline or Pipenet connects multiple models (transformers, detectors, and/or
aggregators) into a "super" model that may perform complex anomaly detection
process.

"""
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle
from tabulate import tabulate

from .._aggregator_base import _Aggregator
from .._base import _Model, _TrainableModel
from .._detector_base import (  # _NonTrainableMultivariateDetector,
    _NonTrainableUnivariateDetector,
    _TrainableMultivariateDetector,
    _TrainableUnivariateDetector,
)
from .._transformer_base import (
    _NonTrainableMultivariateTransformer,
    _NonTrainableUnivariateTransformer,
    _TrainableMultivariateTransformer,
    _TrainableUnivariateTransformer,
)
from ..metrics import f1_score, iou, precision, recall

_Detector = (
    _NonTrainableUnivariateDetector,
    # _NonTrainableMultivariateDetector,
    _TrainableUnivariateDetector,
    _TrainableMultivariateDetector,
)
_Transformer = (
    _NonTrainableUnivariateTransformer,
    _NonTrainableMultivariateTransformer,
    _TrainableUnivariateTransformer,
    _TrainableMultivariateTransformer,
)


class Pipeline:
    """A Pipeline object chains transformers and a detector sequentially.

    Parameters
    ----------
    steps: list of 2-tuples (str, object)
        Components of this pipeline. Each 2-tuple represents a step in the
        pipeline (step name, model object).

    Examples
    --------
    >>> steps = [('moving average', RollingAggregate(agg='mean', window=10)),
                 ('filter quantile 0.99', QuantileAD(high=0.99))]
    >>> myPipeline = Pipeline(steps)

    """

    def __init__(self, steps: List[Tuple[str, _Model]]) -> None:
        self.steps = steps
        self._pipenet = Pipenet()
        self._update_internal_pipenet()

    def _update_internal_pipenet(self) -> None:
        pipenet_steps = dict()
        last_name = "original"
        for pipeline_step in self.steps:
            pipenet_steps.update(
                {
                    pipeline_step[0]: {
                        "model": pipeline_step[1],
                        "input": last_name,
                    }
                }
            )
            last_name = pipeline_step[0]
        self._pipenet.steps = pipenet_steps

    def fit(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
    ) -> Optional[Dict[str, Optional[Union[pd.Series, pd.DataFrame]]]]:
        """Train all models in the pipeline sequentially.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series used to train models.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        dict, optional
            If return_intermediate=True, return intermediate results generated
            during training as a dictionary where keys are step names. If a
            step does not perform transformation or detection, the result of
            that step will be None.

        """
        self._update_internal_pipenet()
        return self._pipenet.fit(
            ts=ts, skip_fit=skip_fit, return_intermediate=return_intermediate
        )

    def detect(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        return_intermediate: bool = False,
        return_list: bool = False,
    ) -> Union[
        Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        Dict[
            str,
            Union[
                pd.Series,
                pd.DataFrame,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
                Dict[
                    str,
                    List[
                        Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
                    ],
                ],
            ],
        ],
    ]:
        """Transform time series sequentially along pipeline, and detect
        anomalies with the last detector.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, list, or dict
            Detected anomalies.

            - If return_intermediate=False, return detected anomalies, i.e.
              result from last detector.
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.
            - If return_list=False, result from a detector or an aggregator
              will be a binary pandas Series indicating normal/anomalous.
            - If return_list=True, result from a detector or an aggregator
              will be a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.


        """
        self._update_internal_pipenet()
        return self._pipenet.detect(
            ts=ts,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def transform(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        return_intermediate: bool = False,
    ) -> Union[
        Union[pd.Series, pd.DataFrame],
        Dict[str, Union[pd.Series, pd.DataFrame]],
    ]:
        """Transform time series sequentially along pipeline.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, or dict
            Transformed time series.

            - If return_intermediate=False, return transformed series, i.e.
              result from last transformer;
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.

        """
        self._update_internal_pipenet()
        return self._pipenet.transform(
            ts=ts, return_intermediate=return_intermediate
        )

    def fit_detect(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
        return_list: bool = False,
    ) -> Union[
        Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        Dict[
            str,
            Union[
                pd.Series,
                pd.DataFrame,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
                Dict[
                    str,
                    List[
                        Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
                    ],
                ],
            ],
        ],
    ]:
        """Train models in pipeline sequentially, transform time series along
        pipeline, and use the last detector to detect anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, list, or dict
            Detected anomalies.

            - If return_intermediate=False, return detected anomalies, i.e.
              result from last detector.
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.
            - If return_list=False, result from a detector or an aggregator
              will be a binary pandas Series indicating normal/anomalous.
            - If return_list=True, result from a detector or an aggregator
              will be a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.


        """
        self._update_internal_pipenet()
        return self._pipenet.fit_detect(
            ts=ts,
            skip_fit=skip_fit,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def fit_transform(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
    ) -> Union[
        Union[pd.Series, pd.DataFrame],
        Dict[str, Union[pd.Series, pd.DataFrame]],
    ]:
        """Train models in pipeline sequentially, and transform time series
        along pipeline.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, or dict
            Transformed time series.

            - If return_intermediate=False, return transformed series, i.e.
              result from last transformer;
            - If return_intermediate=True, return results of all models in
              pipeline as a dict where each item represents the result of a
              model.

        """
        self._update_internal_pipenet()
        return self._pipenet.fit_transform(
            ts=ts, skip_fit=skip_fit, return_intermediate=return_intermediate
        )

    def score(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        anomaly_true: Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        scoring: str = "recall",
        **kwargs: Any
    ) -> Union[float, Dict[str, float]]:
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        anomaly_true: pandas Series or list
            True anomalies.

            - If pandas Series, it is treated as a series of binary labels.
            - If list, a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.

        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou". See module `metrics` for more information.
            Default: "recall"

        **kwargs
            Optional parameters for scoring function. See module `metrics` for
            more information.

        Returns
        -------
        float
            Score of detection result.

        """
        if scoring == "recall":
            scoring_func = recall  # type: Callable
        elif scoring == "precision":
            scoring_func = precision
        elif scoring == "f1":
            scoring_func = f1_score
        elif scoring == "iou":
            scoring_func = iou
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'iou'."
            )
        if isinstance(anomaly_true, pd.Series):
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=False),
                **kwargs
            )
        else:
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=True),
                **kwargs
            )

    def get_params(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters of models in pipeline.

        Returns
        -------
        dict
            A dictionary of model name and model parameters.

        """
        self._update_internal_pipenet()
        return self._pipenet.get_params()


class Pipenet:
    """A Pipenet object connects transformers, detectors and aggregators.

    Parameters
    ----------
    steps: dicts
        Components of the pipenet. Each key-value item represents a step (
        transformer, detector, or aggregator), where key is the unique name of
        the step and the value is a dict with the following key-value pairs:

            - input (str or list of str): Input to the model, which must be
              either 'original' (i.e. the input time series), or the name of
              a upstream component.
            - subset (str, list of str, or list of lists of str, optional): If
              a model does not use all series from an input component, use this
              field to specify which series should be included. If not given or
              "all", all series from the input component will be used.
            - model (object): A detector, transformer, or aggregator object.

    Attributes
    ----------
    steps_graph_: OrderedDict
        Order of steps to be executed. Keys are step names, values are 2-tuple
        (i, j) where i is the index of execution round and j is the the index
        within a round.

    final_step_: str
        Name of the final step to be executed. It is the single step in the
        last round of execution in attribute `steps_graph_`.

    Examples
    --------
    The following example show how to use a Pipenet to build a level shift
    detector with some basic transformers, detectors, and aggregator.

    >>> from adtk.detector import QuantileAD, ThresholdAD
    >>> from adtk.transformer import DoubleRollingAggregate
    >>> from adtk.aggregator import AndAggregator
    >>> from adtk.pipe import Pipenet
    >>> steps = {
            "diff_abs": {
                "input": "original",
                "model": DoubleRollingAggregate(
                    agg="median",
                    window=20,
                    center=True,
                    diff="l1",
                ),
            },
            "quantile_ad": {
                "input": "diff_abs",
                "model": QuantileAD(high=0.99, low=0),
            },
            "diff": {
                "input": "original",
                "model": DoubleRollingAggregate(
                    agg="median",
                    window=20,
                    center=True,
                    diff="diff",
                ),
            },
            "sign_check": {
                "input": "diff",
                "model": ThresholdAD(high=0.0, low=-float("inf")),
            },
            "and": {
                "model": AndAggregator(),
                "input": ["quantile_ad", "sign_check"],
            },
        }
    >>> myPipenet = Pipenet(steps)

    """

    def __init__(
        self, steps: Optional[Dict[str, Dict[str, Any]]] = None
    ) -> None:
        if steps is None:
            self.steps = dict()
        else:
            self.steps = steps
        self._validate()

    def _validate(self) -> None:
        """
        Check the following issues and raise error if found
        - steps is not a list of dict
        - some step does not have required keys
        - some step has invalid keys other than required and optional keys
        - some step's `name` is not str
        - some steps have duplicated name
        - some step uses reserved name
        - some step's `input` is not str or list of str
        - some step's `input` is not a step name or "original"
        - some step's `subset` (optional) is not a str ("all"),
            or a list of (str ("all") or list of str)
        - some step's `subset` does not have the same length as `input`
        - aggregator has a `subset` field
        - some step's `model` is not a valid ADTK model object
        - upstream of a transformer or a detector must be a transformer
        - upstream of aggregator must be a detector or an aggregator

        Create a directed graph of model execution flowchart, when following
        issues will be checked:
        - some step cannot be reached
        - more than one outlet steps found

        """

        # check if step is a dict
        if not isinstance(self.steps, dict):
            raise TypeError("`steps` must be a dict objects.")
        if not all([isinstance(value, dict) for value in self.steps.values()]):
            raise TypeError("Values of dict `steps` must be dict objects.")

        # check if each step has valid keys
        if not all(
            [
                {"model", "input"}
                <= step.keys()
                <= {"model", "input", "subset"}
                for step in self.steps.values()
            ]
        ):
            raise KeyError(
                "Each step must be a dict object with keys `model`, `input`, "
                "and `subset` (optional)."
            )

        # check if each step has valid name
        if not all(
            [isinstance(step_name, str) for step_name in self.steps.keys()]
        ):
            raise TypeError("Model name must be a string.")
        if any([step_name == "original" for step_name in self.steps.keys()]):
            raise ValueError(
                "'original' is a reserved name for original time series input "
                "and you may not use it as a model name."
            )

        # check if each step has valid input
        def islistofstr(li: Any) -> bool:
            if not isinstance(li, list):
                return False
            if not all([isinstance(x, str) for x in li]):
                return False
            return True

        if not all(
            [
                isinstance(step["input"], str) or islistofstr(step["input"])
                for step in self.steps.values()
            ]
        ):
            raise TypeError(
                "Field `input` must be a string or a list of strings."
            )
        name_set = set(self.steps.keys()).union({"original"})
        for step in self.steps.values():
            if (
                isinstance(step["input"], str)
                and (step["input"] not in name_set)
            ) or (
                isinstance(step["input"], list)
                and (not set(step["input"]) <= name_set)
            ):
                raise ValueError(
                    "Field `input` must be 'original' or name of a model."
                )

        # check if only one step is not used as input (so it is the output)
        for step in self.steps.values():
            if isinstance(step["input"], str):
                name_set = name_set - {step["input"]}
            else:
                name_set = name_set - set(step["input"])
        if len(name_set) > 1:
            raise ValueError(
                "Pipenet output is ambiguous: "
                "found more than one steps with no downstream step: {}.".format(
                    name_set
                )
            )

        # check if each step has a valid subset, or has no subset (then we
        # will add default one to it)
        for step_name, step in self.steps.items():
            if isinstance(step["input"], str):
                if "subset" not in step.keys():
                    pass
                elif not (
                    (
                        isinstance(step["subset"], str)
                        or islistofstr(step["subset"])
                    )
                ):
                    raise TypeError(
                        "Field `subset` at step '{}' is invalid.".format(
                            step_name
                        )
                    )
                elif isinstance(step["subset"], str) and (
                    step["subset"] != "all"
                ):
                    raise ValueError(
                        "A subset corresponding to an input source must be "
                        "'all' or a list of strings (even if there is only "
                        "one element)."
                    )
            else:
                if "subset" not in step.keys():
                    pass
                elif isinstance(step["subset"], str):
                    if step["subset"] == "all":
                        pass
                    else:
                        raise ValueError(
                            "Field `subset` at step '{}' is invalid.".format(
                                step_name
                            )
                        )
                elif isinstance(step["subset"], list):
                    if len(step["input"]) != len(step["subset"]):
                        raise ValueError(
                            "Fields `input` and `subset` are inconsistent at "
                            "step '{}'.".format(step_name)
                        )
                    for subset in step["subset"]:
                        if not (
                            (isinstance(subset, str) or islistofstr(subset))
                        ):
                            raise TypeError(
                                "Field `subset` at step '{}' is invalid.".format(
                                    step_name
                                )
                            )
                        if isinstance(subset, str) and (subset != "all"):
                            raise ValueError(
                                "A subset corresponding to an input source "
                                "must be 'all' or a list of strings (even if "
                                "there is only one element)."
                            )
                else:
                    raise TypeError(
                        "Field `subset` at step '{}' is invalid.".format(
                            step_name
                        )
                    )

        # check if each step has valid model
        if not all(
            [isinstance(step["model"], _Model) for step in self.steps.values()]
        ):
            raise ValueError(
                "Model must be a detector, transformer, or aggregator object."
            )

        # check:
        # 1. upstream of transformer and detector must be transformer, or input
        # 2. upstream of aggregator must be detector or aggregator
        for step_name, step in self.steps.items():
            if isinstance(step["model"], (_Detector, _Transformer)):
                if isinstance(step["input"], str):
                    if step["input"] == "original":
                        pass
                    elif not isinstance(
                        self.steps[step["input"]]["model"], _Transformer
                    ):
                        raise TypeError(
                            "Model in step '{}' cannot accept output from "
                            "step '{}'.".format(step_name, step["input"])
                        )
                else:
                    for input in step["input"]:
                        if input == "original":
                            pass
                        elif not isinstance(
                            self.steps[input]["model"], _Transformer
                        ):
                            raise TypeError(
                                "Model in step '{}' cannot accept output from "
                                "step '{}'.".format(step_name, input)
                            )
            elif isinstance(step["model"], _Aggregator):
                if isinstance(step["input"], str):
                    if (step["input"] == "original") or (
                        not isinstance(
                            self.steps[step["input"]]["model"],
                            (_Detector, _Aggregator),
                        )
                    ):
                        raise TypeError(
                            "Model in step '{}' cannot accept output from "
                            "step '{}'.".format(step_name, step["input"])
                        )
                else:
                    for input in step["input"]:
                        if (input == "original") or (
                            not isinstance(
                                self.steps[input]["model"],
                                (_Detector, _Aggregator),
                            )
                        ):
                            raise TypeError(
                                "Model in step '{}' cannot accept output from "
                                "step '{}'.".format(step_name, input)
                            )

        # sort out graph
        done_step = OrderedDict({"original": (0, 0)})
        counter = 0
        while True:
            counter += 1
            sub_counter = 0
            done_step_name_up_to_last_round = set(done_step.keys())
            for step_name, step in self.steps.items():
                if step_name in done_step_name_up_to_last_round:
                    continue
                if isinstance(step["input"], str):
                    if step["input"] in done_step_name_up_to_last_round:
                        done_step.update({step_name: (counter, sub_counter)})
                        sub_counter += 1
                else:
                    if set(step["input"]) <= done_step_name_up_to_last_round:
                        done_step.update({step_name: (counter, sub_counter)})
                        sub_counter += 1
            if len(done_step) == len(self.steps) + 1:
                break
            if sub_counter == 0:
                raise ValueError(
                    "The following step(s) cannot be reached: {}.".format(
                        set(self.steps.keys()) - set(done_step.keys())
                    )
                )
        self.steps_graph_ = done_step.copy()
        self.final_step_ = list(self.steps_graph_.keys())[-1]

    @staticmethod
    def _get_input(
        step: Dict[str, Any], results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Given a step block and an intermediate results dict, get the input of
        this step based on fields `input` and `subset`.
        """
        if isinstance(step["model"], (_Detector, _Transformer)):
            if isinstance(step["input"], str):
                if ("subset" not in step.keys()) or (step["subset"] == "all"):
                    input = results[step["input"]]
                else:
                    if len(step["subset"]) == 1:
                        input = results[step["input"]][step["subset"][0]]
                    else:
                        input = results[step["input"]][step["subset"]]
            else:
                if ("subset" not in step.keys()) or (step["subset"] == "all"):
                    input = pd.concat(
                        [results[input_name] for input_name in step["input"]],
                        axis=1,
                    )
                else:
                    input = pd.concat(
                        [
                            results[input_name]
                            if subset == "all"
                            else (
                                results[input_name][subset[0]]
                                if len(subset) == 1
                                else results[input_name][subset]
                            )
                            for input_name, subset in zip(
                                step["input"], step["subset"]
                            )
                        ],
                        axis=1,
                    )
        elif isinstance(step["model"], _Aggregator):
            if isinstance(step["input"], str):
                input = {step["input"]: results[step["input"]]}
            else:
                input = {
                    input_name: results[input_name]
                    for input_name in step["input"]
                }
        return input

    def fit(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
    ) -> Optional[Dict[str, Optional[Union[pd.Series, pd.DataFrame]]]]:
        """Train models in the pipenet.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series used to train models.

        skip_fit: list, optional
            Models to skip training. This could be used when pipenet contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        dict, optional
            If return_intermediate=True, return intermediate results generated
            during training as a dictionary where keys are step names. If a
            step does not perform transformation or detection, the result of
            that step will be None.

        """
        self._validate()

        if skip_fit is None:
            skip_fit = []
        if not isinstance(skip_fit, list):
            raise TypeError("Parameter `skip_fit` must be a list.")
        if not set(skip_fit) <= set(self.steps.keys()):
            raise ValueError("Name(s) in `skip_fit` is not valid model name.")

        # determine the step needing fit and/or predict
        need_fit = {
            step_name: (
                isinstance(step["model"], _TrainableModel)
                & (step_name not in skip_fit)
            )
            for step_name, step in self.steps.items()
        }
        need_predict = {step_name: False for step_name in self.steps.keys()}
        for step_name in list(self.steps_graph_.keys())[:0:-1]:
            if need_fit[step_name] or need_predict[step_name]:
                if isinstance(self.steps[step_name]["input"], str):
                    input = self.steps[step_name]["input"]
                    if input != "original":
                        need_predict[input] = True
                else:
                    for input in self.steps[step_name]["input"]:
                        if input != "original":
                            need_predict[input] = True

        # run fit or fit_predict
        results = {"original": ts.copy()}
        for step_name in list(self.steps_graph_.keys())[1:]:
            step = self.steps[step_name]
            if not (need_fit[step_name] or need_predict[step_name]):
                results.update({step_name: None})
                continue
            input = self._get_input(step, results)
            if need_fit[step_name] and (not need_predict[step_name]):
                step["model"].fit(input)
                results.update({step_name: None})
            elif (not need_fit[step_name]) and need_predict[step_name]:
                results.update({step_name: step["model"].predict(input)})
            else:
                results.update({step_name: step["model"].fit_predict(input)})

        # return intermediate results
        if return_intermediate:
            return results
        else:
            return None

    def _predict(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        fit: bool,
        detect: bool,
        skip_fit: Optional[List[str]],
        return_intermediate: bool,
        return_list: bool = False,
    ) -> Union[
        Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        Dict[
            str,
            Union[
                pd.Series,
                pd.DataFrame,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
                Dict[
                    str,
                    List[
                        Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
                    ],
                ],
            ],
        ],
    ]:
        """
        Private method for detect, transform, fit_detect, fit_transform

        Parameters
        ----------
        fit: bool
            Whether this call is for fit_detect/fit_transform or
            detect/transform.

        detect: bool
            Whether this call is for detect/fit_detect or
            transform/fit_transform.

        Others:
            Same as higher-level calls

        Returns
        -------
            Same as higher-level calls

        """
        self._validate()

        if skip_fit is None:
            skip_fit = []
        if not isinstance(skip_fit, list):
            raise TypeError("Parameter `skip_fit` must be a list.")
        if not set(skip_fit) <= set(self.steps.keys()):
            raise ValueError("Name(s) in `skip_fit` is not valid model name.")

        last_step_name = list(self.steps_graph_.keys())[-1]

        if detect:
            if isinstance(self.steps[last_step_name]["model"], _Transformer):
                raise RuntimeError(
                    "This seems a transformation pipenet, "
                    "because model at the final step '{}' is a transformer. "
                    "Please use method `{}transform` instead.".format(
                        last_step_name, "fit_" if fit else ""
                    )
                )
        else:
            if isinstance(
                self.steps[last_step_name]["model"], (_Detector, _Aggregator)
            ):
                raise RuntimeError(
                    "This seems a detection pipenet, "
                    "because model at the final step '{}' is "
                    "either a detector or an aggregator. "
                    "Please use method `{}detect` instead.".format(
                        last_step_name, "fit_" if fit else ""
                    )
                )

        results = {
            "original": ts.copy()
        }  # type: Dict[str,Union[pd.Series,pd.DataFrame,List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]], Dict[str, List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]]]]]
        for step_name in list(self.steps_graph_.keys())[1:]:
            step = self.steps[step_name]
            input = self._get_input(step, results)
            results.update(
                {
                    step_name: (
                        (
                            step["model"].fit_predict(
                                input, return_list=return_list
                            )
                            if isinstance(
                                step["model"],
                                (
                                    _TrainableUnivariateDetector,
                                    _TrainableMultivariateDetector,
                                ),
                            )
                            else step["model"].predict(
                                input, return_list=return_list
                            )
                        )
                        if isinstance(step["model"], _Detector)
                        else (
                            step["model"].fit_predict(input)
                            if isinstance(
                                step["model"],
                                (
                                    _TrainableUnivariateTransformer,
                                    _TrainableMultivariateTransformer,
                                ),
                            )
                            else step["model"].predict(input)
                        )
                    )
                    if fit and (step_name not in skip_fit)
                    else (
                        step["model"].predict(input, return_list=return_list)
                        if isinstance(step["model"], _Detector)
                        else step["model"].predict(input)
                    )
                }
            )

        if return_intermediate:
            return results
        else:
            return results[last_step_name]

    def detect(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        return_intermediate: bool = False,
        return_list: bool = False,
    ) -> Union[
        Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        Dict[
            str,
            Union[
                pd.Series,
                pd.DataFrame,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
                Dict[
                    str,
                    List[
                        Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
                    ],
                ],
            ],
        ],
    ]:
        """Detect anomaly from time series using the pipenet.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, list, or dict
            Detected anomalies.

            - If return_intermediate=False, return detected anomalies, i.e.
              result from last detector.
            - If return_intermediate=True, return results of all models in
              pipenet as a dict where each item represents the result of a
              model.
            - If return_list=False, result from a detector or an aggregator
              will be a binary pandas Series indicating normal/anomalous.
            - If return_list=True, result from a detector or an aggregator
              will be a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.

        """
        return self._predict(
            ts,
            fit=False,
            detect=True,
            skip_fit=None,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def transform(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        return_intermediate: bool = False,
    ) -> Union[
        Union[pd.Series, pd.DataFrame],
        Dict[str, Union[pd.Series, pd.DataFrame]],
    ]:
        """Transform time series using the pipenet.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, or dict
            Transformed time series.

            - If return_intermediate=False, return transformed series, i.e.
              result from last transformer;
            - If return_intermediate=True, return results of all models in
              pipnet as a dict where each item represents the result of a
              model.

        """
        return self._predict(
            ts,
            fit=False,
            detect=False,
            skip_fit=None,
            return_intermediate=return_intermediate,
        )

    def fit_detect(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
        return_list: bool = False,
    ) -> Union[
        Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        Dict[
            str,
            Union[
                pd.Series,
                pd.DataFrame,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
                Dict[
                    str,
                    List[
                        Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]
                    ],
                ],
            ],
        ],
    ]:
        """Train models in the pipenet and detect anomaly with it.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        skip_fit: list, optional
            Models to skip training. This could be used when pipenet contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous events, or a binary series
            indicating normal/anomalous. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, list, or dict
            Detected anomalies.

            - If return_intermediate=False, return detected anomalies, i.e.
              result from last detector.
            - If return_intermediate=True, return results of all models in
              pipenet as a dict where each item represents the result of a
              model.
            - If return_list=False, result from a detector or an aggregator
              will be a binary pandas Series indicating normal/anomalous.
            - If return_list=True, result from a detector or an aggregator
              will be a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.

        """
        return self._predict(
            ts,
            fit=True,
            detect=True,
            skip_fit=skip_fit,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def fit_transform(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        skip_fit: Optional[List[str]] = None,
        return_intermediate: bool = False,
    ) -> Union[
        Union[pd.Series, pd.DataFrame],
        Dict[str, Union[pd.Series, pd.DataFrame]],
    ]:
        """Train models in the pipenet and transform time series with it.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed.

        skip_fit: list, optional
            Models to skip training. This could be used when pipenet contains
            models that are already trained by the same time series, and re-
            training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        pandas Series, pandas DataFrame, or dict
            Transformed time series.

            - If return_intermediate=False, return transformed series, i.e.
              result from last transformer;
            - If return_intermediate=True, return results of all models in
              pipenet as a dict where each item represents the result of a
              model.


        """
        return self._predict(
            ts,
            fit=True,
            detect=False,
            skip_fit=skip_fit,
            return_intermediate=return_intermediate,
        )

    def score(
        self,
        ts: Union[pd.Series, pd.DataFrame],
        anomaly_true: Union[
            pd.Series,
            pd.DataFrame,
            List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            Dict[
                str,
                List[Union[Tuple[pd.Timestamp, pd.Timestamp], pd.Timestamp]],
            ],
        ],
        scoring: str = "recall",
        **kwargs: Any
    ) -> Union[float, Dict[str, float]]:
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        anomaly_true: Series, or a list of Timestamps or Timestamp tuple
            True anomalies.

            - If pandas Series, it is treated as a series of binary labels.
            - If list, a list of events where an event is a pandas Timestamp if
              it is instantaneous or a 2-tuple of pandas Timestamps if it is a
              closed time interval.

        scoring: str, optional
            Scoring function to use. Must be one of "recall", "precision",
            "f1", and "iou". See module `metrics` for more information.
            Default: "recall"

        **kwargs
            Optional parameters for scoring function. See module `metrics` for
            more information.

        Returns
        -------
        float
            Score of detection result.

        """
        if scoring == "recall":
            scoring_func = recall  # type: Callable
        elif scoring == "precision":
            scoring_func = precision
        elif scoring == "f1":
            scoring_func = f1_score
        elif scoring == "iou":
            scoring_func = iou
        else:
            raise ValueError(
                "Argument `scoring` must be one of 'recall', 'precision', "
                "'f1' and 'iou'."
            )
        if isinstance(anomaly_true, pd.Series):
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=False),
                **kwargs
            )
        else:
            return scoring_func(
                y_true=anomaly_true,
                y_pred=self.detect(ts, return_list=True),
                **kwargs
            )

    def get_params(self) -> Dict[str, Dict[str, Any]]:
        """Get parameters of models in pipenet.

        Returns
        -------
        dict
            A dictionary of model name and model parameters.

        """
        return {
            step_name: step["model"].get_params()
            for step_name, step in self.steps.items()
        }

    def summary(self) -> None:
        """Print a summary of the pipenet."""
        df = pd.DataFrame(columns=["name", "model", "input", "subset"])
        for step_name in self.steps_graph_.keys():
            if step_name == "original":
                continue
            df = df.append(
                {
                    "name": step_name,
                    "model": self.steps[step_name]["model"].__class__.__name__,
                    "input": self.steps[step_name]["input"],
                    "subset": (
                        self.steps[step_name]["subset"]
                        if "subset" in self.steps[step_name].keys()
                        else "all"
                    ),
                },
                ignore_index=True,
            )
        print(tabulate(df, headers="keys", tablefmt="simple", showindex=False))

    def plot_flowchart(
        self,
        ax: Optional[plt.Axes] = None,
        figsize: Optional[Tuple[float, float]] = None,
        radius: float = 1.0,
    ) -> plt.Axes:  # pragma: no cover
        """Plot flowchart of this pipenet.

        Parameters
        ----------
        ax: matplotlib axes object, optional
            Axes to plot at. If not given, the method will create a matplotlib
            figure and axes. Default: None.

        figsize: tuple, optional
            Width and height of the figure to plot at. Only to be used if `ax`
            is not given. Default: None.

        radius: float, optional
            Relative size of components in the chart. Default: 1.0.

        Returns
        -------
        matplotlib axes object
            Axes where the flowchart is plotted.

        """
        self._validate()

        radius = radius * 0.1

        # create empty plot
        if (ax is None) and (figsize is None):
            figsize = (14, 16)
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        # get coordinate of components
        layers = []  # type: List
        for graph_step, values in self.steps_graph_.items():
            if values[1] == 0:
                layers.append([])
            layers[-1].append(graph_step)
        n_layer = len(layers)
        max_n_comp = 0
        coord = dict()
        for layer in layers:
            n_comp = len(layer)
            max_n_comp = max(max_n_comp, n_comp)
            for comp in layer:
                x = self.steps_graph_[comp][0]
                y = self.steps_graph_[comp][1] - (n_comp - 1) / 2
                coord[comp] = (x, y)

        # plot connection lines, and gather patches
        io_patches = []  # type: List
        detector_patches = []  # type: List
        transformer_patches = []  # type: List
        aggregator_patches = []  # type: List
        for step_name, step in self.steps.items():
            end_coord = coord[step_name]
            if isinstance(step["model"], _Detector):
                detector_patches.append(Circle(xy=end_coord, radius=radius))
            elif isinstance(step["model"], _Transformer):
                transformer_patches.append(Circle(xy=end_coord, radius=radius))
            elif isinstance(step["model"], _Aggregator):
                aggregator_patches.append(Circle(xy=end_coord, radius=radius))
            if isinstance(step["input"], str):
                input = step["input"]
                start_coord = coord[input]
                if ("subset" not in step.keys()) or (step["subset"] == "all"):
                    linestyle = "-"
                else:
                    linestyle = "--"
                plt.plot(
                    [start_coord[0], end_coord[0]],
                    [start_coord[1], end_coord[1]],
                    color="k",
                    linestyle=linestyle,
                    zorder=1,
                )
            else:
                for counter, input in enumerate(step["input"]):
                    start_coord = coord[input]
                    if (
                        ("subset" not in step.keys())
                        or (step["subset"] == "all")
                        or (step["subset"][counter] == "all")
                    ):
                        linestyle = "-"
                    else:
                        linestyle = "--"
                    plt.plot(
                        [start_coord[0], end_coord[0]],
                        [start_coord[1], end_coord[1]],
                        color="k",
                        linestyle=linestyle,
                        zorder=1,
                    )
        plt.plot([n_layer - 1, n_layer], [0, 0], color="k", zorder=1)
        io_patches.append(Circle(xy=(0, 0), radius=radius))
        io_patches.append(Circle(xy=(n_layer, 0), radius=radius))

        # draw components
        ax.add_collection(PatchCollection(io_patches, zorder=2, color="y"))
        ax.add_collection(
            PatchCollection(detector_patches, zorder=2, color="g")
        )
        ax.add_collection(
            PatchCollection(transformer_patches, zorder=2, color="c")
        )
        ax.add_collection(
            PatchCollection(aggregator_patches, zorder=2, color="m")
        )

        # label components with step names
        for key, value in coord.items():
            ax.annotate(
                key, (value[0] - radius, value[1]), zorder=3, color="k"
            )
        ax.annotate("result", (n_layer - radius, 0), zorder=3, color="k")

        # add legend
        plt.legend(
            handles=[
                Line2D([], [], color="k", label="full connection"),
                Line2D(
                    [],
                    [],
                    color="k",
                    linestyle="--",
                    label="partial connection",
                ),
                Circle([], radius=radius, color="y", label="input/output"),
                Circle([], radius=radius, color="g", label="detector"),
                Circle([], radius=radius, color="c", label="transformer"),
                Circle([], radius=radius, color="m", label="aggregator"),
            ],
            bbox_to_anchor=(1, 1),
            loc=2,
            borderaxespad=0.0,
        )

        # clean up axes
        ax.set_xlim([-radius - 0.1, n_layer + radius + 0.1])
        ax.set_ylim(
            [
                -(max_n_comp - 1) / 2 - radius - 0.1,
                (max_n_comp - 1) / 2 + radius + 0.1,
            ]
        )
        ax.set_axis_off()
        ax.set_aspect(1)

        return ax
