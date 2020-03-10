================================
Anomaly Detection Toolkit (ADTK)
================================

Anomaly Detection Toolkit (ADTK) is a Python package for unsupervised /
rule-based time series anomaly detection.

As the nature of anomaly varies over different cases, a model may not work
universally for all anomaly detection problems. Choosing and combining
detection algorithms (detectors), feature engineering methods (transformers),
and ensemble methods (aggregators) properly is the key to build an effective
anomaly detection model.

This package offers a set of common detectors, transformers and aggregators
with unified APIs, as well as pipe classes that connect them together into a
model. It also provides some functions to process and visualize time series and
anomaly events.

.. include::
   install.rst

.. include::
   quickstart.rst

.. toctree::
   :caption: Table of Contents
   :maxdepth: 1

   install
   quickstart
   userguide
   examples
   api/modules
   developer
   releasehistory


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
