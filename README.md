# Anomaly Detection Toolkit (ADTK)

[![Build Status](https://travis-ci.com/arundo/adtk.svg?branch=master)](https://travis-ci.com/arundo/adtk)
[![Documentation Status](https://readthedocs.org/projects/adtk/badge/?version=stable)](https://adtk.readthedocs.io/en/stable)
[![Coverage Status](https://coveralls.io/repos/github/arundo/adtk/badge.svg?branch=master&service=github)](https://coveralls.io/github/arundo/adtk?branch=master)
[![PyPI](https://img.shields.io/pypi/v/adtk)](https://pypi.org/project/adtk/)
[![Downloads](https://pepy.tech/badge/adtk)](https://pepy.tech/project/adtk)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/arundo/adtk/master?filepath=docs%2Fnotebooks%2Fdemo.ipynb)

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

See https://adtk.readthedocs.io for complete documentation.

## Installation

Prerequisites: Python 3.5 or later.

It is recommended to install the most recent **stable** release of ADTK from PyPI.

```shell
pip install adtk
```

Alternatively, you could install from source code. This will give you the **latest**, but unstable, version of ADTK.

```shell
git clone https://github.com/arundo/adtk.git
cd adtk/
git checkout develop
pip install ./
```

## Examples

Please see [Quick Start](https://adtk.readthedocs.io/en/stable/quickstart.html) for a simple example.

For more detailed examples of each module of ADTK, please refer to
[Examples](https://adtk.readthedocs.io/en/stable/examples.html)
section in the documentation or [an interactive demo notebook](https://mybinder.org/v2/gh/arundo/adtk/master?filepath=docs%2Fnotebooks%2Fdemo.ipynb).

## Contributing

Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update unit tests as appropriate.

Please see [Contributing](https://adtk.readthedocs.io/en/stable/developer.html) for more details.


## License

ADTK is licensed under the Mozilla Public License 2.0 (MPL 2.0). See the
[LICENSE](LICENSE) file for details.
