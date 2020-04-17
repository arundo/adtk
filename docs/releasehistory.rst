***************
Release History
***************

Version 0.6.2 (Apr 16, 2020)
===================================
- Hot fix of wrong documentation url

Version 0.6.1 (Apr 16, 2020)
===================================
- Migrated the documentation to a new host
- Fixed minor typos in the documentation
- Fixed a minor type hinting bug

Version 0.6.0 (Mar 10, 2020)
===================================
- Re-designed the API of :py:mod:`adtk.visualization.plot`
- Removed :py:mod:`adtk.data.resample` because its functionality is highly overlapped with pandas resampler module
- Made :py:mod:`adtk.data.expand_events` accept events in the form of pandas Series/DataFrame
- Made :py:mod:`adtk.data.expand_events` accept time delta in the form of `str` or `int`
- Changed the output type of :py:mod:`adtk.data.split_train_test` from a 2-tuple of lists to a list of 2-tuples
- Turned the following model parameters required from optional

    - `window` in :py:mod:`adtk.detector.LevelShiftAD`
    - `window` in :py:mod:`adtk.detector.VolatilityShiftAD`
    - `window` in :py:mod:`adtk.transformer.RollingAggregate`
    - `window` in :py:mod:`adtk.transformer.DoubleRollingAggregate`
    - `model` in :py:mod:`adtk.detector.MinClusterDetector`
    - `model` in :py:mod:`adtk.detector.OutlierDetector`
    - `target` and `regressor` in :py:mod:`adtk.detector.RegressionAD`
    - `target` and `regressor` in :py:mod:`adtk.transformer.RegressionResidual`
    - `aggregate_func` in :py:mod:`adtk.aggregator.CustomizedAggregator`
    - `detect_func` in :py:mod:`adtk.detector.CustomizedDetector1D`
    - `detect_func` in :py:mod:`adtk.detector.CustomizedDetectorHD`
    - `transform_func` in :py:mod:`adtk.transformer.CustomizedTransformer1D`
    - `transform_func` in :py:mod:`adtk.detector.CustomizedTransformer1D`
    - `steps` in :py:mod:`adtk.pipe.Pipeline`

- Added consistency check between training and testing inputs in multivariate models
- Improved time index check in time-dependent models
- Turned all second-order sub-modules private, and a user now can only import from the following first-order modules

    - :py:mod:`adtk.detector`
    - :py:mod:`adtk.transformer`
    - :py:mod:`adtk.aggregator`
    - :py:mod:`adtk.pipe`
    - :py:mod:`adtk.data`
    - :py:mod:`adtk.metrics`
    - :py:mod:`adtk.visualization`

- Refactored the inheritance structure of model components (see :ref:`inheritance`)
- Added Python 3.8 support
- Fixed compatibility issues with statsmodels v0.11
- Fixed compatibility issues with pandas v1.0
- Created an interactive demo notebook in Binder
- Added type hints, and added type checking in CI/CD test
- Added `Black` and `isort` to developer requirement and CI/CD check
- Optimized release process by publishing package to PyPI through GitHub Actions
- Improved docstrings and API documentation
- Fixed many minor bugs and typos

Version 0.5.5 (Feb 24, 2020)
===================================
- Fixed a bug that empty lists were ignored by AndAggregator
- Fixed some typo in the documentation

Version 0.5.4 (Feb 18, 2020)
===================================
- Optimized the workflow of how a univariate model is applied to pandas DataFrame

    - Added more informative error messages
    - Fixed some bugs resulting in model-column matching error due to inconsistency between output Series names and DataFrame columns
    - Clarified the workflow in the documentation

Version 0.5.3 (Feb 12, 2020)
===================================
- Quick hotfix to avoid errors caused by statsmodels v0.11 by requiring statsmodels dependency <0.11

Version 0.5.2 (Jan 14, 2020)
===================================
- Formalized the management of releases and pre-releases, including rules of branches and versioning
- Added more rules for developers to the documentation

Version 0.5.1 (Jan 2, 2020)
===================================
- Added many new unit tests, and modified some old unit test
- Removed seaborn from dependencies (use matplotlib built-in style now)
- Fixed a bug in the metric module of dict objects as input
- Fixed a bug in the detector OutlierDetector that output series has dtype object if NaN is present
- Fixed a bug in transformer pipeline that detect and transform methods are confused
- Fixed a bug in pipenet that an aggregator node may crash if its input is from a node where subset contains a single item
- Fixed a bug in pipenet summary that subset column are always "all" even if not
- Some minor optimization of code

Version 0.5.0 (Dec 18, 2019)
===================================
- Changed the parameter `steps` of pipenet from list to dict
- Added method `summary` to pipenet
- Corrected some major algorithmic issues on seasonal decomposition

    - Removed STL decomposition transformer, and hence the corresponding option in SeasonalAD detector
    - Recreated classic seasonal decomposition transformer

- Updated the demo notebook in the documentation
- Added an option to hide legend in the plotting function
- Added some package setup options for developers
- Fixed an issue of tracking Travis and Coveralls status
- Some minor internal optimization in the code
- Fixed some format issues and typos in the documentation

Version 0.4.1 (Nov 21, 2019)
===================================
- Fixed an issue of tox environments
- Minor spelling/grammar fix in documentation

Version 0.4.0 (Nov 18, 2019)
===================================
- Added support to Python 3.5
- Better unit tests on dependencies
- Minor typo fix in documentation
- Minor code optimization
- Added download statistics to README
- Added coverage test

Version 0.3.0 (Sep 27, 2019)
===================================
- Initial release