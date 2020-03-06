"""Module of model pipeline and pipenet.

Pipeline or Pipenet connects multiple components (transformers, detectors,
and/or aggregators) into a model that may perform complex anomaly detection
process.

"""

from ._pipe import Pipeline, Pipenet

__all__ = ["Pipeline", "Pipenet"]
