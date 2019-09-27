"""
Anomaly Detection Toolkit
=========================

Anomaly Detection Toolkit (ADTK) is a Python package for unsupervised /
rule-based time series anomaly detection.

As the nature of anomaly varies over different cases, a model may not work
universally for all anomaly detection problems. Choosing and combining
detection algorithms (detectors), feature engineering methods (transformers),
and ensemble methods (aggregators) properly is the key to build an effective
anomaly detection model.

This package offers a set of common detectors, transformers and aggregators
with unified APIs, as well as pipe classes that connect them together into
models. It also provides some functions to process and visualize time series
and anomaly events.

See https://arundo-adtk.readthedocs-hosted.com for complete documentation.

"""

# PEP0440 compatible formatted version, see:
# https://www.python.org/dev/peps/pep-0440/
#
# Generic release markers:
#   X.Y
#   X.Y.Z   # For bugfix releases
#
# Admissible pre-release markers:
#   X.YaN   # Alpha release
#   X.YbN   # Beta release
#   X.YrcN  # Release Candidate
#   X.Y     # Final release
#
# Dev branch marker is: 'X.Y.dev' or 'X.Y.devN' where N is an integer.
# 'X.Y.dev0' is the canonical version of 'X.Y.dev'
#
__version__ = "0.3"
