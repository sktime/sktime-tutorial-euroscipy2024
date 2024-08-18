# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""
Implementation of ResidualBoostingForecaster.

A forecaster that fits base forecasters to the residuals of the previous
forecasters.
"""

__author__ = ["felipeangelimvieira"]
__all__ = ["ResidualBoostingForecaster"]

from typing import Optional

import pandas as pd
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
    def __init__(
        self, base_forecaster: Optional[BaseForecaster] = None, num_iter: int = 2
    ):
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
            [   "capability:exogenous",
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
        cumulative_preds = pd.Series(0, index=y.index)
        for _ in range(self.num_iter):
            forecaster = self._base_forecaster.clone()
            forecaster.fit(timeseries_to_predict, X, fh)

            # Forecast insample
            insample_fh = y.index.get_level_values(-1).unique()
            insample_preds = forecaster.predict(fh=insample_fh, X=X)
            cumulative_preds += insample_preds.values
            # Get residuals
            negative_residuals = y - cumulative_preds.values
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

    def _update(self, y, X=None, update_params=True):
        """Update time series to incremental training data.

        private _update containing the core logic, called from update

        State required:
            Requires state to be "fitted".

        Accesses in self:
            Fitted model attributes ending in "_"
            self.cutoff

        Writes to self:
            Sets fitted model attributes ending in "_", if update_params=True.
            Does not write to self if update_params=False.

        Parameters
        ----------
        y : sktime time series object
            guaranteed to be of an mtype in self.get_tag("y_inner_mtype")
            Time series with which to update the forecaster.
            if self.get_tag("scitype:y")=="univariate":
                guaranteed to have a single column/variable
            if self.get_tag("scitype:y")=="multivariate":
                guaranteed to have 2 or more columns
            if self.get_tag("scitype:y")=="both": no restrictions apply
        X :  sktime time series object, optional (default=None)
            guaranteed to be of an mtype in self.get_tag("X_inner_mtype")
            Exogeneous time series for the forecast
        update_params : bool, optional (default=True)
            whether model parameters should be updated

        Returns
        -------
        self : reference to self
        """
        
        timeseries_to_predict = y.copy()
        cumulative_preds = pd.Series(0, index=y.index)
        for forecaster in self.forecasters_:
            
            forecaster.update(timeseries_to_predict, X, update_params=update_params)

            # Forecast insample
            insample_fh = y.index.get_level_values(-1).unique()
            insample_preds = forecaster.predict(fh=insample_fh, X=X)
            cumulative_preds += insample_preds.values
            # Get residuals
            negative_residuals = y - cumulative_preds.values
            timeseries_to_predict = negative_residuals
            

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

        return [
            {
                "base_forecaster": None,
                "num_iter": 1
            },
            {
                "base_forecaster": ExponentialSmoothing(),
                "num_iter": 10,
            },
        ]
