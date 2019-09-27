# Anomaly Detection Toolkit (ADTK)

[![Build Status](https://travis-ci.com/arundo/adtk.svg?branch=master)](https://travis-ci.com/arundo/adtk)
[![Docs](https://readthedocs.com/projects/arundo-adtk/badge/?version=latest)](https://arundo-adtk.readthedocs-hosted.com/en/latest/)
[![PyPI](https://img.shields.io/pypi/v/adtk)](https://pypi.org/project/adtk/)

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

## Installation

Prerequisites: Python 3.6 or later.

It is recommended to use **pip** for installation.

```shell
pip install adtk
```

Alternatively, you could install from source code:

```shell
git clone https://github.com/arundo/adtk.git
cd adtk/
pip install ./
```

## Examples

Please see [Quick Start](https://arundo-adtk.readthedocs-hosted.com/en/latest/quickstart.html) for a simple example.

For more detailed examples of each module of ADTK, please refer to
[Examples](https://arundo-adtk.readthedocs-hosted.com/en/latest/examples.html)
section in the documentation.

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

ADTK is licensed under the Mozilla Public License 2.0 (MPL 2.0). See the
[LICENSE](LICENSE) file for details.
