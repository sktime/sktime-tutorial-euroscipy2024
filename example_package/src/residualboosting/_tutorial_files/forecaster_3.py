# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Implementation of ResidualBoostingForecaster.

A forecaster that fits base forecasters to the residuals of the previous
forecasters.
"""

__author__ = ["felipeangelimvieira"]

from typing import Optional

from sktime.forecasting.base import BaseForecaster
from sktime.forecasting.exp_smoothing import ExponentialSmoothing


class ResidualBoostingForecaster(BaseForecaster):
    """Residual Boosting Forecaster.

    This forecaster uses a base forecaster to fit the residuals of the previous
    base forecaster instances, until

    Parameters
    ----------
    forecaster : sktime forecaster
        The base forecaster instance to use.
    num_iter: int, default=2
        Number of forecasters to fit to the residuals of the previous forecasters.
        Should be at least 1
    """

    _tags = {
        # Tags that we won't clone from base forecaster
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "scitype:y": "univariate",
        "capability:pred_int": False,
        "capability:pred_int:insample": False,
        "capability:insample": True,
        # Tags to clone
        "ignores-exogeneous-X": False,
        "requires-fh-in-fit": True,
        "X-y-must-have-same-index": True,
        "enforce_index_type": None,
        "handles-missing-data": False,
        "authors": ["felipeangelimvieira"],
        "maintainers": ["felipeangelimvieira"],
        "python_version": None,
        "python_dependencies": None,
    }

    # todo: add any hyper-parameters and components to constructor
    def __init__(self, base_forecaster: Optional[BaseForecaster] = None, num_iter: int = 2):
        # estimators should precede parameters
        #  if estimators have default values, set None and initialize below

        self.base_forecaster = base_forecaster
        self.num_iter = num_iter

        super().__init__()

        # Handle default base forecaster
        if base_forecaster is None:
            self._base_forecaster = ExponentialSmoothing()
        else:
            self._base_forecaster = base_forecaster

        # Parameter checking logic
        if not self._base_forecaster.get_tag("capability:insample"):
            raise ValueError("Base forecaster must have capability:insample")

        if num_iter < 1:
            raise ValueError("num_iter must be at least 1")

        # if tags of estimator depend on component tags, set them
        self.clone_tags(
            self._base_forecaster,
            [
                "ignores-exogeneous-X",
                "requires-fh-in-fit",
                "X-y-must-have-same-index",
                "enforce_index_type",
                "handles-missing-data",
                "python_version",  # TODO: Should clone python version and dpendencies?
                "python_dependencies",
            ],
        )

    def _fit(self, y, X, fh):
        """Fit forecaster to training data.

        private _fit containing the core logic, called from fit

        Writes to self:
            Sets fitted model attributes ending in "_".

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of pd.Series
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : sktime time series object, optional (default=None)
            guaranteed to be pd.DataFrame, or None
            

        Returns
        -------
        self : reference to self
        """

        timeseries_to_predict = y.copy()
        self.forecasters_ = []
        for i in range(self.num_iter):
            forecaster = self._base_forecaster.clone()
            forecaster.fit(timeseries_to_predict, X, fh)

            # Forecast insample
            insample_fh = y.index.get_level_values(-1).unique()
            insample_preds = forecaster.predict(fh=insample_fh, X=X)

            # Get residuals
            negative_residuals = y - insample_preds
            timeseries_to_predict = negative_residuals
            self.forecasters_.append(forecaster)

        return self

    
    def _predict(self, fh, X):
        """Forecast time series at future horizon.

        private _predict containing the core logic, called from predict

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Parameters
        ----------
        fh : ForecastingHorizon or None, optional (default=None)
            The forecasting horizon with the steps ahead to to predict.
            Required (non-optional) here if self.get_tag("requires-fh-in-fit")==True
            Otherwise, if not passed in _fit, guaranteed to be passed in _predict
        X : sktime time series object, optional (default=None)
            guaranteed to be pd.DataFrame, or None

        Returns
        -------
        y_pred : sktime time series object
            should be of the same type as seen in _fit, as in "y_inner_mtype" tag
            Point predictions
        """
        y_pred = 0
        for forecaster in self.forecasters_:
            y_pred += forecaster.predict(fh, X)
        
        return y_pred

    @classmethod
    def get_test_params(cls, parameter_set="default"):
        """Return testing parameter settings for the estimator.

        Parameters
        ----------
        parameter_set : str, default="default"
            Name of the set of test parameters to return, for use in tests. If no
            special parameters are defined for a value, will return `"default"` set.
            There are currently no reserved values for forecasters.

        Returns
        -------
        params : dict or list of dict, default = {}
            Parameters to create testing instances of the class
            Each dict are parameters to construct an "interesting" test instance, i.e.,
            `MyClass(**params)` or `MyClass(**params[i])` creates a valid test instance.
            `create_test_instance` uses the first (or only) dictionary in `params`
        """

        # todo: set the testing parameters for the estimators
        # Testing parameters can be dictionary or list of dictionaries
        # Testing parameter choice should cover internal cases well.
        #
        # this method can, if required, use:
        #   class properties (e.g., inherited); parent class test case
        #   imported objects such as estimators from sktime or sklearn
        # important: all such imports should be *inside get_test_params*, not at the top
        #            since imports are used only at testing time
        #
        # The parameter_set argument is not used for automated, module level tests.
        #   It can be used in custom, estimator specific tests, for "special" settings.
        # A parameter dictionary must be returned *for all values* of parameter_set,
        #   i.e., "parameter_set not available" errors should never be raised.
        #
        # A good parameter set should primarily satisfy two criteria,
        #   1. Chosen set of parameters should have a low testing time,
        #      ideally in the magnitude of few seconds for the entire test suite.
        #       This is vital for the cases where default values result in
        #       "big" models which not only increases test time but also
        #       run into the risk of test workers crashing.
        #   2. There should be a minimum two such parameter sets with different
        #      sets of values to ensure a wide range of code coverage is provided.
        #
        # example 1: specify params as dictionary
        # any number of params can be specified
        # params = {"est": value0, "parama": value1, "paramb": value2}
        #
        # example 2: specify params as list of dictionary
        # note: Only first dictionary will be used by create_test_instance
        # params = [{"est": value1, "parama": value2},
        #           {"est": value3, "parama": value4}]
        # return params
        #
        # example 3: parameter set depending on param_set value
        #   note: only needed if a separate parameter set is needed in tests
        # if parameter_set == "special_param_set":
        #     params = {"est": value1, "parama": value2}
        #     return params
        #
        # # "default" params - always returned except for "special_param_set" value
        # params = {"est": value3, "parama": value4}
        # return params
