.. _inheritance:

Model Classes Inheritance Diagram
==================================

    .. code-block:: console

        _Model
            |-- _NonTrainableModel
            |       |-- _NonTrainableUnivariateModel
            |       |       |-- _NonTrainableUnivariateDetector
            |       |       |       └-- ThresholdAD
            |       |       |
            |       |       └-- _NonTrainableUnivariateTransformer
            |       |               |-- RollingAggregate
            |       |               |-- DoubleRollingAggregate
            |       |               |-- Retrospect
            |       |               └-- StandardScale
            |       |
            |       └-- _NonTrainableMultivariateModel
            |               └-- _NonTrainableMultivariateTransformer
            |                       └-- SumAll
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
            |       |       |       └-- CustomizedDetector1D
            |       |       |
            |       |       └-- _TrainableUnivariateTransformer
            |       |               |-- ClassicSeasonalDecomposition
            |       |               └-- CustomizedTransformer1D
            |       |
            |       └-- _TrainableMultivariateModel
            |               |-- _TrainableMultivariateDetector
            |               |       |-- MinClusterDetector
            |               |       |-- OutlierDetector
            |               |       |-- RegressionAD
            |               |       |-- PcaAD
            |               |       └-- CustomizedDetectorHD
            |               |
            |               └-- _TrainableMultivariateTransformer
            |                       |-- RegressionResidual
            |                       |-- PcaProjection
            |                       |-- PcaReconstruction
            |                       |-- PcaReconstructionError
            |                       └-- CustomizedTransformerHD
            |
            └-- _Aggregator
                    |-- AndAggregator
                    |-- OrAggregator
                    └-- CustomizedAggregator
