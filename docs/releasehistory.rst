***************
Release History
***************

Version 0.6.0-dev
===================================
- Support str and int as time delta for the input arguments in functions `expand_events` and `resample` in the data module (0.6.0-dev.1+pr.39)
- Added an example of DoubleRollingAggregate with different window sizes to the documentation (0.6.0-dev.5+pr.43)
- Optimized release process by publishing package to PyPI through GitHub Actions (0.6.0-dev.7+pr.54, 0.6.0-dev.8+pr.56)

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