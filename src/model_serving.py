'''Functions used for model training, inference, and evaluation. '''
import datetime
from typing import Union
import sys

import sktime
import pandas as pd
import numpy as np
import pandas
import datetime
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.compose import RecursiveTabularRegressionForecaster

sys.path.append("../src")
from eda import ts_lineplot_facet_row

class JetModels:
    def __init__(self, train_set: pandas.DataFrame,
                 test_set: pandas.DataFrame,
                 error_metric: sktime.performance_metrics):
        '''
        Wrapper class that allows for fitting a sktime forecaster to a train set and a test set and provides several
        utlities, such as fit plotting and error calculation.

        Args:
            train_set (pandas.DataFrame): The train set, use multiindex to create hierarchical forecasts
            test_set (pandas.DataFrame): The test set, use multiindex to create hierarchical forecasts
            error_metric (sktime.performance_metrics): A sktime.performance_metrics class or function to use for
                evaluation.
        '''
        # Check that train test set have both same index type
        if type(test_set.index) != type(train_set.index):
            raise TypeError(f"Index types vary between train and test set!")

        # Assign class attributes
        self.train_set = train_set
        self.test_set = test_set
        self.error_metric = error_metric

        # Determine forecasting horizon for test set and window size for CV strategy based if multiindex is present
        if isinstance(self.train_set.index, pandas.MultiIndex):
            # assumes that last index is time as required by sktime
            self.initial_window_size = self.train_set\
                .groupby(self.train_set.index.names[:-1]).size().min()

            # All test sets need to have the same group size
            len_fh = self.test_set.groupby(self.test_set.index.names[:-1]).size()

            # Raise value error if they aren't
            if len_fh.min() != len_fh.max():
                raise ValueError(f"Not all test sets have the same size!")
            self.fh_test = ForecastingHorizon(self.test_set.index.get_level_values(level=-1).unique(),
                                               is_relative=False)
            self.len_fh = len_fh.min()

            # We must do the same for generating training predictions, but have to use the actual index there
            self.fh_train = ForecastingHorizon(train_set.index.get_level_values(-1).unique(), is_relative=False)
        else:
            self.initial_window_size = len(self.train_set)
            self.fh_test = ForecastingHorizon(self.test_set.index,
                                              is_relative=False)
            self.fh_train = ForecastingHorizon(train_set.index, is_relative=False)

    def simple_forecast(self,
                        forecaster: sktime.forecasting,
                        fh_individual: Union[np.ndarray, list] = None):
        '''
        Trains the forecaster on the train set and performs inference on test set.

        Args:
            forecaster (sktime.forecasting): A sktime forecaster object that will be used as model.
            fh_individual(Union[np.ndarray, list], optional): Individual forecasting horizon to use, otherwise uses
                test_set length as forecast horizon.

        Returns:
            (tuple): A tuple consisting of the fitted forecaster, the train predictions, and the test set predictions
        '''
        # Fit forecaster
        forecaster.fit(self.train_set)

        # For RecursiveTabularRegressionForecaster the in-sample prediction is not yet implemented. This is a crude
        # hotfix
        if isinstance(forecaster, RecursiveTabularRegressionForecaster):
            print("In sample prediction for RecursiveTabularRegressionForecaster not implemented. "
                  "Returning train set as predictions.")
            train_preds = self.train_set
        else:
            # Get train predictions
            train_preds = forecaster.predict(self.fh_train)

        # Perform forecasting on either test_set length or individual forecasting horizon
        if fh_individual is not None:
            fh_to_use = fh_individual
        else:
            fh_to_use = self.fh_test

        forecast_vals = forecaster.predict(fh=fh_to_use)

        # Round predicted values to integer
        forecast_vals = forecast_vals.round(0).astype(int)

        result = (forecaster, train_preds, forecast_vals)

        return result

    def cv_forecast(self,
                    search_strat_class: sktime.forecasting.model_selection,
                    fh_individual: Union[np.ndarray, list] = None):
        '''
        Trains forecaster using specified cv nd search strategy. For selected model, train predictions, as well as
        forecasting values are provided, either using test_set length or specified forecasting horizon.
        Args:
            search_strat_class (sktime.forecasting.model_selection): Instantiated sktime.forecasting.model_selection
                class that governs how optimal model, i.e. hyperparameter selection strategy. Examples include
                ForecastingGridSearchCV
            fh_individual(Union[np.ndarray, list], optional): Individual forecasting horizon to use, otherwise uses
                test_set length as forecast horizon.
        Returns:
            (tuple): A tuple consisting of the fitted forecaster, the train predictions, and the test set predictions
        '''
        # Perform cross-validation, hyperparameter tuning, and fit model
        search_strat_class.fit(self.train_set)

        # For RecursiveTabularRegressionForecaster the in-sample prediction is not yet implemented. This is a crude
        # hotfix
        if isinstance(search_strat_class.forecaster, RecursiveTabularRegressionForecaster):
            print("In sample prediction for RecursiveTabularRegressionForecaster not implemented. "
                  "Returning train set as predictions.")
            train_preds = self.train_set
        else:
            # Get train predictions
            train_preds = search_strat_class.predict(self.fh_train)

        # Print hyperparameters selected
        print(f"Model fitted. Best params: {search_strat_class.best_params_}")

        # Perform forecasting on either test_set length or individual forecasting horizon
        if fh_individual is not None:
            fh_to_use = fh_individual
        else:
            fh_to_use = self.fh_test

        forecast_vals = search_strat_class.predict(fh=fh_to_use)

        # Round predicted values to integer
        forecast_vals = forecast_vals.round(0).astype(int)

        result = (search_strat_class, train_preds, forecast_vals)

        return result

    def plot_error_curves(self,
                          test_vals: pandas.Series,
                          train_vals: pandas.Series,
                          train_horizon: int = 14,
                          width: int = 800,
                          height: int = 1200,
                          title: str = "",
                          facet_row_wrap: int = 0,
                          facet_row_spacing: float = 0.06):
        '''
        Produces training and test fit plots based on class specified error metric.

        In case of multiindex of train or test set (must both be either multi or single index) will assume a
        hierarchical data structure and use all level up to the last one as grouping to facet plots.

        Args:
            test_vals (pandas.Series): Pandas series of forecasted values for the test set
            train_vals (pandas.Series): Pandas series of forecasted values for the train set
            train_horizon (int, optional): How many days (per group if present) before start of test set to include in
                plots. Assumes that train and test sets where split temporally.
            width (int, optional): width of plot
            height (int, optional): height of plot
            title (str, optional): Title to use for the plot
            facet_row_wrap (int, optional): Maximum number of facet rows. Ignored if 0.
            facet_row_spacing (float, optional):  Spacing between facet rows, in paper units.

        Returns:
            None
        '''
        # Copy train and test set to prevent argument alteration
        train_set_used = self.train_set.copy()
        test_set_used = self.test_set.copy()

        # Check that train test have both same index type
        if type(test_vals.index) != type(train_vals.index):
            raise TypeError(f"Index types vary between train and test set predictions!")

        # Check if series all have the same index names
        test_preds_names = {*test_vals.index.names}
        train_preds_names = {*train_vals.index.names}
        test_set_names = {*self.test_set.index.names}
        train_set_names = {*self.train_set.index.names}

        all_sets_equal = (test_preds_names == train_preds_names == test_set_names == train_set_names)

        # technically would need to check target column naming as well..

        if not all_sets_equal:
            raise KeyError("Not all index names of predictions and ground truth sets are the same!")

        # First check if multi index is present, if so, no grouping will be applied, otherwise uses levels coming before
        # last one as grouping
        if isinstance(test_vals.index, pandas.MultiIndex):
            # Names are by above check the same
            grouping = test_vals.index.names[:-1] # last one is assumed to be time index
        else:
            grouping = None

        time_index_col = test_vals.index.names[-1]

        # Get start and end date based on time column for plot data frame and user input
        start_date_test = self.test_set.reset_index()[time_index_col].dt.date.min()
        end_date = self.test_set.reset_index()[time_index_col].dt.date.max()
        start_date = start_date_test - datetime.timedelta(days=train_horizon)

        # Add train / test identifier
        train_vals["error_type"] = "train"
        test_vals["error_type"] = "test"
        train_set_used["error_type"] = "train"
        test_set_used["error_type"] = "test"

        # We now create a new data frame that contains the predictions in the correct format to use plotly (long-format)
        # First create a prediction and ground truth data frame, then combine into on, after adding identifier variable
        predictions = pd.concat([train_vals, test_vals], axis=0)
        predictions["data_type"] = "prediction"
        truth = pd.concat([train_set_used, test_set_used], axis=0)
        truth["data_type"] = "ground_truth"

        plot_df = pd.concat([predictions.reset_index(), truth.reset_index()], axis=0, ignore_index=True)

        # Subset to only relevant time points
        mask = (plot_df[time_index_col].dt.date >= start_date) & (plot_df[time_index_col].dt.date <= end_date)
        plot_df = plot_df.loc[mask, :]

        # Create plot, either faceted if multiindex or simple
        if grouping is not None:
            fig = ts_lineplot_facet_row(plot_df,
                              x_var = time_index_col,
                              y_var = "ridership",
                              color = "data_type",
                              width = width,
                              height = height,
                              title = title,
                              facet_row = grouping[-1],
                              facet_row_spacing=facet_row_spacing)

        else:
            fig = ts_lineplot_facet_row(plot_df,
                                    x_var=time_index_col,
                                    y_var="ridership",
                                    color="data_type",
                                    width=width,
                                    height=height,
                                    title=title)
        return fig

    def calculate_grouped_errors(self,
                         test_vals: pandas.Series):
        '''
        Calculates forecast errors based on error_metric attribute.

        In case of multiindex of test vals will assume a hierarchical data structure and use all level up to the last
        one as grouping to calculate grouped error values.

        Args:
            test_vals (pandas.Series): Pandas series of forecasted values for the test set

        Returns:
            (pandas.DataFrame): A pandas DataFrame with one column corresponding to the grouping and one to the error
                values for that group.
        '''
        # Check if multiindex
        if not isinstance(test_vals.index, pandas.MultiIndex):
            raise ValueError("Supplied series has not multiindex! Use the error metric directly for ungrouped series.")

        # Empty list to store values
        error_vals = []
        groups = test_vals.index.get_level_values(level=0).unique().values
        group_name = test_vals.index.names[0]

        # Loop through each group and get error values
        for group in groups:
            error_value = self.error_metric(
                y_true = self.test_set.loc[group, "ridership"],
                y_pred = test_vals.loc[group, "ridership"],
                y_train = self.train_set.loc[group, "ridership"]
            )
            error_vals.append(error_value)

        # Generate output Data Frame
        output = pd.DataFrame({group_name :groups, "error_value" : error_vals})

        return output