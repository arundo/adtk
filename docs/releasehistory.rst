***************
Release History
***************

Version 0.6.0-dev
===================================
- Support `str` and `int` as time delta for the input arguments in functions `expand_events` and `resample` in the data module (0.6.0-dev.1+pr.39)
- Added an example of DoubleRollingAggregate with different window sizes to the documentation (0.6.0-dev.5+pr.43)
- Optimized release process by publishing package to PyPI through GitHub Actions (0.6.0-dev.7+pr.54, 0.6.0-dev.8+pr.56, 0.6.0-dev.9+pr.57, 0.6.0-dev.10+pr.58)
- Created an interactive demo notebook in Binder (0.6.0-dev.12+pr.64)
- Fixed compatibility issues with statsmodels v0.11 (0.6.0-dev.15+pr.72)
- Fixed compatibility issues with pandas v1.0 (0.6.0-dev.16+pr.73)
- Added Python 3.8 support (0.6.0-dev.17+pr.74)
- Added type hints, and added type checking in CI/CD test (0.6.0-dev.19+pr.79)
- Refactored the inheritance structure (0.6.0-dev.19-pr.79)

    .. code-block:: console

        _Model
            |-- _NonTrainableModel
            |       |-- _NonTrainableUnivariateModel
            |       |       |-- _NonTrainableUnivariateDetector
            |       |       |       |-- ThresholdAD
            |       |       |
            |       |       |-- _NonTrainableUnivariateTransformer
            |       |               |-- RollingAggregate
            |       |               |-- DoubleRollingAggregate
            |       |               |-- Retrospect
            |       |               |-- StandardScale
            |       |
            |       |-- _NonTrainableMultivariateModel
            |               |-- _NonTrainableMultivariateTransformer
            |                       |-- SumAll
            |
            |-- _TrainableModel
            |       |-- _TrainableUnivariateModel
            |       |       |-- _TrainableUnivariateDetector
            |       |       |       |-- QuantileAD
            |       |       |       |-- InterQuartileRangeAD
            |       |       |       |-- GeneralizedESDTestAD
            |       |       |       |-- PersistAD
            |       |       |       |-- LevelShiftAD
            |       |       |       |-- VolatilityShiftAD
            |       |       |       |-- SeasonalAD
            |       |       |       |-- AutoregressionAD
            |       |       |       |-- CustomizedDetector1D
            |       |       |
            |       |       |-- _TrainableUnivariateTransformer
            |       |               |-- ClassicSeasonalDecomposition
            |       |               |-- CustomizedTransformer1D
            |       |
            |       |-- _TrainableMultivariateModel
            |               |-- _TrainableMultivariateDetector
            |               |       |-- MinClusterDetector
            |               |       |-- OutlierDetector
            |               |       |-- RegressionAD
            |               |       |-- PcaAD
            |               |       |-- CustomizedDetectorHD
            |               |
            |               |-- _TrainableMultivariateTransformer
            |                       |-- RegressionResidual
            |                       |-- PcaProjection
            |                       |-- PcaReconstruction
            |                       |-- PcaReconstructionError
            |                       |-- CustomizedTransformerHD
            |
            |-- _Aggregator
                    |-- AndAggregator
                    |-- OrAggregator
                    |-- CustomizedAggregator

- We made all second-order sub-modules private and user now can only import from first-order modules (0.6.0-dev.19-pr.79)

    - adtk.detector
    - adtk.transformer
    - adtk.aggregator
    - adtk.pipe
    - adtk.data
    - adtk.metrics
    - adtk.visualization

- Improved docstrings and API documentation (0.6.0-dev.19-pr.79)
- Fixed minor bugs and typos (0.6.0-dev.19-pr.79)
- Turned some parameters in some models required (0.6.0-dev.19-pr.79)

    - `window` in `adtk.detector.LevelShiftAD`
    - `window` in `adtk.detector.VolatilityShiftAD`
    - `window` in `adtk.transformer.RollingAggregate`
    - `window` in `adtk.transformer.DoubleRollingAggregate`
    - `model` in `adtk.detector.MinClusterDetector`
    - `model` in `adtk.detector.OutlierDetector`
    - `target` and `regressor` in `adtk.detector.RegressionAD`
    - `target` and `regressor` in `adtk.transformer.RegressionResidual`
    - `aggregate_func` in `adtk.aggregator.CustomizedAggregator`
    - `detect_func` in `adtk.detector.CustomizedDetector1D`
    - `detect_func` in `adtk.detector.CustomizedDetectorHD`
    - `transform_func` in `adtk.transformer.CustomizedTransformer1D`
    - `transform_func` in `adtk.detector.CustomizedTransformer1D`
    - `steps` in `adtk.pipe.Pipeline`

- Re-designed the API of `adtk.visualization.plot` (0.6.0-dev.20-pr.80)
- Added `Black` and `isort` to developer requirement and CI/CD check (0.6.0-dev.21-pr.88)

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