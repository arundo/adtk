"""Module for model pipeline and pipenet.

Pipeline or Pipenet connects multiple models (transformers, detectors, and/or
aggregators) into a "super" model that may perform complex anomaly detection
process.

"""
from collections import OrderedDict

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D
from matplotlib.patches import Circle

from .._base import _Model
from .._detector_base import _Detector1D, _DetectorHD
from .._transformer_base import _Transformer1D, _TransformerHD
from .._aggregator_base import _Aggregator
from ..metrics import recall, precision, f1_score, iou


__all__ = ["Pipeline", "Pipenet"]


class Pipeline:
    """A Pipeline object chains transformers and a detector sequentially.

    Parameters
    ----------
    steps: list of 2-tuples
        Components of this pipeline. Each 2-tuple represents a step in the
        pipeline (step name, model object).

    Examples
    --------
    >>> steps = [('moving average',
                  RollingAggregate(agg='mean', window=10)),
                 ('filter quantile 0.99',
                  QuantileAD(upper_thresh=0.99))]
    >>> myPipeline = Pipeline(steps)

    """

    def __init__(self, steps=None):
        if steps is None:
            self.steps = []
        else:
            self.steps = steps
        self._pipenet = Pipenet()
        self._update_internal_pipenet()

    def _update_internal_pipenet(self):
        pipenet_steps = [
            {"name": pipeline_step[0], "model": pipeline_step[1]}
            for pipeline_step in self.steps
        ]
        pipenet_steps[0].update({"input": "original"})
        for this, previous in zip(pipenet_steps[1:], pipenet_steps[:-1]):
            this.update({"input": previous["name"]})
        self._pipenet.steps = pipenet_steps

    def fit(self, ts, skip_fit=None, return_intermediate=False):
        """Train all models in the pipeline sequentially.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series used to train models.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and
            re-training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        dict
            If return_intermediate=True, return intermediate results generated
            during training as a dictionary where keys are step names. If a
            step does not perform transformation or detection, the result of
            that step will be None.

        """
        self._update_internal_pipenet()
        return self._pipenet.fit(
            ts=ts, skip_fit=skip_fit, return_intermediate=return_intermediate
        )

    def detect(self, ts, return_intermediate=False, return_list=False):
        """Transform time series sequentially along pipeline, and detect
        anomalies with the last detector.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        list, panda Series, or dict
            If return_intermediate=False, return detected anomalies, i.e.
            result from last detector;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipeline.
            If return_list=True, result from a detector or an aggregators will
            be a list of pandas Timestamps;
            If return_list=False, result from a detector or an aggregators will
            be a binary pandas Series indicating normal/anomalous.


        """
        self._update_internal_pipenet()
        return self._pipenet.detect(
            ts=ts,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def transform(self, ts, return_intermediate=False):
        """Transform time series sequentially along pipeline.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        list or dict
            If return_intermediate=False, return transformed dataframe, i.e.
            result from last transformer;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipeline.

        """
        self._update_internal_pipenet()
        return self._pipenet.detect(
            ts=ts, return_intermediate=return_intermediate
        )

    def fit_detect(
        self, ts, skip_fit=None, return_intermediate=False, return_list=False
    ):
        """Train models in pipeline sequentially, transform time series along
        pipeline, and use the last detector to detect anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and
            re-training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        list, panda Series, or dict
            If return_intermediate=False, return detected anomalies, i.e.
            result from last detector;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipeline.
            If return_list=True, result from a detector or an aggregators will
            be a list of pandas Timestamps;
            If return_list=False, result from a detector or an aggregators will
            be a binary pandas Series indicating normal/anomalous.


        """
        self._update_internal_pipenet()
        return self._pipenet.fit_detect(
            ts=ts,
            skip_fit=skip_fit,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def fit_transform(self, ts, skip_fit=None, return_intermediate=False):
        """Train models in pipeline sequentially, and transform time series
        along pipeline.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed

        skip_fit: list, optional
            Models to skip training. This could be used when pipeline contains
            models that are already trained by the same time series, and
            re-training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        list or dict
            If return_intermediate=False, return transformed dataframe, i.e.
            result from last transformer;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipeline.

        """
        self._update_internal_pipenet()
        return self._pipenet.fit_transform(
            ts=ts, skip_fit=skip_fit, return_intermediate=return_intermediate
        )

    def score(self, ts, anomaly_true, scoring="recall", **kwargs):
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, k univariate detectors will be
            applied to them respectively.

        anomaly_true: Series, or a list of Timestamps or Timestamp tuple
            True anomalies.
            If Series, it is a series binary labels indicating anomalous;
            If list, it is a list of anomalous events in form of time windows.

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
            scoring_func = recall
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

    def get_params(self):
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
    steps: list of dicts
        Components of the pipenet. Each dict represents a model (transformer,
        detector, or aggregator), and has four key values:

            - name (str): A unique name of this components.
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

    >>> from adtk.detector_1d import QuantileAD, ThresholdAD
    >>> from adtk.transformer_1d import DoubleRollingAggregate
    >>> from adtk.aggregator import AndAggregator
    >>> from adtk.pipe import Pipenet
    >>> steps = [
            {
                "name": "diff_abs",
                "input": "original",
                "model": DoubleRollingAggregate(
                    agg="median",
                    window=20,
                    center=True,
                    diff="l1",
                ),
            },
            {
                "name": "quantile_ad",
                "input": "diff_abs",
                "model": QuantileAD(upper_thresh=0.99, lower_thresh=0),
            },
            {
                "name": "diff",
                "input": "original",
                "model": DoubleRollingAggregate(
                    agg="median",
                    window=20,
                    center=True,
                    diff="diff",
                ),
                "input": "original",
            },
            {
                "name": "sign_check",
                "input": "diff",
                "model": ThresholdAD(upper_thresh=0.0, lower_thresh=-float("inf")),
            },
            {
                "name": "and",
                "model": AndAggregator(),
                "input": ["quantile_ad", "sign_check"],
            },
        ]
    >>> myPipenet = Pipenet(steps)

    """

    def __init__(self, steps=None):
        if steps is None:
            self.steps = []
        else:
            self.steps = steps
        self._validate()

    def _validate(self):
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

        # check if step is a list of dict
        if not isinstance(self.steps, list):
            raise TypeError("`steps` must be a list of dict objects.")
        if not all([isinstance(step, dict) for step in self.steps]):
            raise TypeError("`steps` must be a list of dict objects.")

        # check if each step has valid keys
        if not all(
            [
                {"name", "model", "input"}
                <= step.keys()
                <= {"name", "model", "input", "subset"}
                for step in self.steps
            ]
        ):
            raise KeyError(
                "Each step must be a dict object with keys `name`, `model`, "
                "`input`, and `subset` (optional)."
            )

        # check if each step has valid name
        if not all([isinstance(step["name"], str) for step in self.steps]):
            raise TypeError("Model name must be a string.")
        if len({step["name"] for step in self.steps}) < len(self.steps):
            raise ValueError("Model names must be unique.")
        if any([step["name"] == "original" for step in self.steps]):
            raise ValueError(
                "'original' is a reserved name for original time series input "
                "and you may not use it as a model name."
            )

        # check if each step has valid input
        def islistofstr(li):
            if not isinstance(li, list):
                return False
            if not all([isinstance(x, str) for x in li]):
                return False
            return True

        if not all(
            [
                isinstance(step["input"], str) or islistofstr(step["input"])
                for step in self.steps
            ]
        ):
            raise TypeError(
                "Field `input` must be a string or a list of strings."
            )
        name_set = {step["name"] for step in self.steps}.union({"original"})
        for step in self.steps:
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
        for step in self.steps:
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
        for step in self.steps:
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
                            step["name"]
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
                                step["name"]
                            )
                        )
                elif isinstance(step["subset"], list):
                    if len(step["input"]) != len(step["subset"]):
                        raise ValueError(
                            "Fields `input` and `subset` are inconsistent at "
                            "step '{}'.".format(step["name"])
                        )
                    for subset in step["subset"]:
                        if not (
                            (isinstance(subset, str) or islistofstr(subset))
                        ):
                            raise TypeError(
                                "Field `subset` at step '{}' is invalid.".format(
                                    step["name"]
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
                            step["name"]
                        )
                    )

        # check if each step has valid model
        if not all([isinstance(step["model"], _Model) for step in self.steps]):
            raise ValueError(
                "Model must be a detector, transformer, or aggregator object."
            )

        # check:
        # 1. upstream of transformer and detector must be transformer, or input
        # 2. upstream of aggregator must be detector or aggregator
        name2ind = {
            step["name"]: counter for counter, step in enumerate(self.steps)
        }
        for step in self.steps:
            if isinstance(
                step["model"],
                (_Detector1D, _DetectorHD, _Transformer1D, _TransformerHD),
            ):
                if isinstance(step["input"], str):
                    if step["input"] == "original":
                        pass
                    elif not isinstance(
                        self.steps[name2ind[step["input"]]]["model"],
                        (_Transformer1D, _TransformerHD),
                    ):
                        raise TypeError(
                            "Model in step '{}' cannot accept output from "
                            "step '{}'.".format(step["name"], step["input"])
                        )
                else:
                    for input in step["input"]:
                        if input == "original":
                            pass
                        elif not isinstance(
                            self.steps[name2ind[input]]["model"],
                            (_Transformer1D, _TransformerHD),
                        ):
                            raise TypeError(
                                "Model in step '{}' cannot accept output from "
                                "step '{}'.".format(step["name"], input)
                            )
            elif isinstance(step["model"], _Aggregator):
                if isinstance(step["input"], str):
                    if (step["input"] == "original") or (
                        not isinstance(
                            self.steps[name2ind[step["input"]]]["model"],
                            (_Detector1D, _DetectorHD, _Aggregator),
                        )
                    ):
                        raise TypeError(
                            "Model in step '{}' cannot accept output from "
                            "step '{}'.".format(step["name"], step["input"])
                        )
                else:
                    for input in step["input"]:
                        if (input == "original") or (
                            not isinstance(
                                self.steps[name2ind[input]]["model"],
                                (_Detector1D, _DetectorHD, _Aggregator),
                            )
                        ):
                            raise TypeError(
                                "Model in step '{}' cannot accept output from "
                                "step '{}'.".format(step["name"], input)
                            )

        # sort out graph
        done_step = OrderedDict({"original": (0, 0)})
        counter = 0
        while True:
            counter += 1
            sub_counter = 0
            done_step_name_up_to_last_round = set(done_step.keys())
            for step in self.steps:
                if step["name"] in done_step_name_up_to_last_round:
                    continue
                if isinstance(step["input"], str):
                    if step["input"] in done_step_name_up_to_last_round:
                        done_step.update(
                            {step["name"]: (counter, sub_counter)}
                        )
                        sub_counter += 1
                else:
                    if set(step["input"]) <= done_step_name_up_to_last_round:
                        done_step.update(
                            {step["name"]: (counter, sub_counter)}
                        )
                        sub_counter += 1
            if len(done_step) == len(self.steps) + 1:
                break
            if sub_counter == 0:
                raise ValueError(
                    "The following step(s) cannot be reached: {}.".format(
                        {step["name"] for step in self.steps}
                        - set(done_step.keys())
                    )
                )
        self.steps_graph_ = done_step.copy()
        self.final_step_ = list(self.steps_graph_.keys())[-1]

    @staticmethod
    def _get_input(step, results):
        """
        Given a step block and an intermediate results dict, get the input of
        this step based on fields `input` and `subset`.
        """
        if isinstance(
            step["model"],
            (_Detector1D, _DetectorHD, _Transformer1D, _TransformerHD),
        ):
            if isinstance(step["input"], str):
                if ("subset" not in step.keys()) or (step["subset"] == "all"):
                    input = results[step["input"]]
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
                            else results[input_name][subset]
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

    def fit(self, ts, skip_fit=None, return_intermediate=False):
        """Train models in the pipenet.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series used to train models.

        skip_fit: list, optional
            Models to skip training. This could be used when pipenet contains
            models that are already trained by the same time series, and
            re-training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        dict
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
        if not set(skip_fit) <= {step["name"] for step in self.steps}:
            raise ValueError("Name(s) in `skip_fit` is not valid model name.")

        # determine the step needing fit and/or predict
        need_fit = {
            step["name"]: (
                (step["model"]._need_fit) & (step["name"] not in skip_fit)
            )
            for step in self.steps
        }
        need_predict = {step["name"]: False for step in self.steps}
        name2ind = {
            step["name"]: counter for counter, step in enumerate(self.steps)
        }
        for step_name in list(self.steps_graph_.keys())[:0:-1]:
            if need_fit[step_name] or need_predict[step_name]:
                if isinstance(self.steps[name2ind[step_name]]["input"], str):
                    input = self.steps[name2ind[step_name]]["input"]
                    if input != "original":
                        need_predict[input] = True
                else:
                    for input in self.steps[name2ind[step_name]]["input"]:
                        if input != "original":
                            need_predict[input] = True

        # run fit or fit_predict
        results = {"original": ts.copy()}
        for step_name in list(self.steps_graph_.keys())[1:]:
            step = self.steps[name2ind[step_name]]
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

    def _predict(
        self, ts, fit, detect, skip_fit, return_intermediate, return_list
    ):
        """
        Private method for detect, transform, fit_detect, fit_transform
        """
        self._validate()

        if skip_fit is None:
            skip_fit = []
        if not isinstance(skip_fit, list):
            raise TypeError("Parameter `skip_fit` must be a list.")
        if not set(skip_fit) <= {step["name"] for step in self.steps}:
            raise ValueError("Name(s) in `skip_fit` is not valid model name.")

        name2ind = {
            step["name"]: counter for counter, step in enumerate(self.steps)
        }
        last_step_name = list(self.steps_graph_.keys())[-1]

        if detect:
            if isinstance(
                self.steps[name2ind[last_step_name]]["model"],
                (_Transformer1D, _TransformerHD),
            ):
                raise RuntimeError(
                    "This seems a transformation pipenet, "
                    "because model at the final step '{}' is a transformer. "
                    "Please use method `{}transform` instead.".format(
                        last_step_name, "fit_" if fit else ""
                    )
                )
        else:
            if isinstance(
                self.steps[name2ind[last_step_name]]["model"],
                (_Detector1D, _DetectorHD, _Aggregator),
            ):
                raise RuntimeError(
                    "This seems a detection pipenet, "
                    "because model at the final step '{}' is "
                    "either a detector or an aggregator. "
                    "Please use method `{}detect` instead.".format(
                        last_step_name, "fit_" if fit else ""
                    )
                )

        results = {"original": ts.copy()}
        for step_name in list(self.steps_graph_.keys())[1:]:
            step = self.steps[name2ind[step_name]]
            input = self._get_input(step, results)
            results.update(
                {
                    step_name: step["model"].fit_predict(
                        input, return_list=return_list
                    )
                    if fit and (step_name not in skip_fit)
                    else step["model"].predict(input, return_list=return_list)
                }
            )

        if return_intermediate:
            return results
        else:
            return results[last_step_name]

    def detect(self, ts, return_intermediate=False, return_list=False):
        """Detect anomaly from time series using the pipenet.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        list, panda Series, or dict
            If return_intermediate=False, return detected anomalies, i.e.
            result from last detector;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipenet.
            If return_list=True, result from a detector or an aggregators will
            be a list of pandas Timestamps;
            If return_list=False, result from a detector or an aggregators will
            be a binary pandas Series indicating normal/anomalous.

        """
        return self._predict(
            ts,
            fit=False,
            detect=True,
            skip_fit=None,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def transform(self, ts, return_intermediate=False):
        """Transform time series using the pipenet.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        list or dict
            If return_intermediate=False, return transformed dataframe, i.e.
            result from last transformer;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipenet.

        """
        return self._predict(
            ts,
            fit=False,
            detect=False,
            skip_fit=None,
            return_intermediate=return_intermediate,
            return_list=None,
        )

    def fit_detect(
        self, ts, skip_fit=None, return_intermediate=False, return_list=False
    ):
        """Train models in the pipenet and detect anomaly with it.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.

        skip_fit: list, optional
            Models to skip training. This could be used when pipenet contains
            models that are already trained by the same time series, and
            re-training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        return_list: bool, optional
            Whether to return a list of anomalous time stamps, or a binary
            series indicating normal/anomalous. Default: False.

        Returns
        -------
        list, panda Series, or dict
            If return_intermediate=False, return detected anomalies, i.e.
            result from last detector;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipenet.
            If return_list=True, result from a detector or an aggregators will
            be a list of pandas Timestamps;
            If return_list=False, result from a detector or an aggregators will
            be a binary pandas Series indicating normal/anomalous.

        """
        return self._predict(
            ts,
            fit=True,
            detect=True,
            skip_fit=skip_fit,
            return_intermediate=return_intermediate,
            return_list=return_list,
        )

    def fit_transform(self, ts, skip_fit=None, return_intermediate=False):
        """Train models in the pipenet and transform time series with it.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to be transformed.

        skip_fit: list, optional
            Models to skip training. This could be used when pipenet contains
            models that are already trained by the same time series, and
            re-training would be time consuming. It must be a list of strings
            where each element is a model name. Default: None.

        return_intermediate: bool, optional
            Whether to return intermediate results. Default: False.

        Returns
        -------
        list or dict
            If return_intermediate=False, return transformed dataframe, i.e.
            result from last transformer;
            If return_intermediate=True, return a dictionary of model results
            for all models in pipenet.

        """
        return self._predict(
            ts,
            fit=True,
            detect=False,
            skip_fit=skip_fit,
            return_intermediate=return_intermediate,
            return_list=None,
        )

    def score(self, ts, anomaly_true, scoring="recall", **kwargs):
        """Detect anomalies and score the results against true anomalies.

        Parameters
        ----------
        ts: pandas Series or DataFrame
            Time series to detect anomalies from.
            If a DataFrame with k columns, k univariate detectors will be
            applied to them respectively.

        anomaly_true: Series, or a list of Timestamps or Timestamp tuple
            True anomalies.
            If Series, it is a series binary labels indicating anomalous;
            If list, it is a list of anomalous events in form of time windows.

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
            scoring_func = recall
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

    def get_params(self):
        """Get parameters of models in pipenet.

        Returns
        -------
        dict
            A dictionary of model name and model parameters.

        """
        return {
            step["name"]: step["model"].get_params() for step in self.steps
        }

    def plot_flowchart(self, ax=None, figsize=None, radius=1.0):
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
        layers = []
        for step, values in self.steps_graph_.items():
            if values[1] == 0:
                layers.append([])
            layers[-1].append(step)
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
        io_patches = []
        detector_patches = []
        transformer_patches = []
        aggregator_patches = []
        for step in self.steps:
            end_coord = coord[step["name"]]
            if isinstance(step["model"], (_Detector1D, _DetectorHD)):
                detector_patches.append(Circle(xy=end_coord, radius=radius))
            elif isinstance(step["model"], (_Transformer1D, _TransformerHD)):
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
