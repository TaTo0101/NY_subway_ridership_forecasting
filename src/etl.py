### Functions that are used for extract, load, transform operations ###

import os
from typing import Union
import pandas as pd
import numpy as np
import datetime
import warnings
class DataLoaderRidership:
    def __init__(self, path:str,
                 time_col: str = "transit_timestamp",
                 dtype_dict: dict = None,
                 date_format: str = None,
                 add_cols: dict = {"rider_col" : "ridership",
                                   "station_id" : "station_complex_id",
                                   "fare_class" : "fare_class_category"}):
        '''
        Creates a DataLoaderRidership class, that handles data loading and type conversions for the MTA ridership data.

        CSV file specified must adhere to the format obtained when exporting data from
        https://data.ny.gov/Transportation/MTA-Subway-Hourly-Ridership-Beginning-February-202/wujg-7c2s

        Args:
            path (str): The path to the csv file
            time_col (str): The column name that contains the time identifier, default is transit_timestamp
            dtype_dict (dict, optional): A dictionary of the format column_name : dtype, where dtype specifies the
                dtype of the column. If not specified pandas will guess the dtypes (not memory efficient)
            date_format (str, optional): The date format for the column specified in time_col. If not provided not date
                conversion will take place.
            add_cols (dict, optional): A dictionary that specifies the column names for ridership, station id, and fare
                class.
        '''
        # Assign attributes
        self.path = path
        self.time_col = time_col # must be a list for pandas.read_csv
        self.dtype_dict = dtype_dict
        self.date_format = date_format
        self.ridership_col = add_cols["rider_col"]
        self.station_id_col = add_cols["station_id"]
        self.fare_class_col = add_cols["fare_class"]
        self.add_cols = {self.ridership_col, self.station_id_col, self.fare_class_col}

        # Check if provided path exists
        if not os.path.isfile(self.path):
            raise ValueError(f"Provided path {self.path} does not contain a valid file!")

        # Check that time col exists
        with open(self.path) as f:
            column_names = f.read().strip("\n").split(",")
        if time_col not in column_names:
            raise KeyError(f"Specified time_col {time_col} is not a variable in csv file. If default was not changed "
                             f"this could hint at the fact that the csv provided is incompatible.")
        # Check that columns specified in add_col exist
        missing_add_cols = self.add_cols.difference({*column_names})
        if len(missing_add_cols)>0:
            raise KeyError(f"Some additional columns don't exist in csv file! Columns that are missing:\n"
                           f"{missing_add_cols}")

        # Load data
        self.data_load()

        # Add time variables
        self.add_time_variables()

        # Add daily ridership
        self.add_daily_ridership()

    def data_load(self):
        '''
        Loads the csv file and if specified performs type conversions, e.g. date conversion.

        Returns:
            None
        '''
        df = pd.read_csv(self.path, dtype=self.dtype_dict, parse_dates=[self.time_col], date_format=self.date_format)

        self.df = df

    def add_time_variables(self):
        '''
        Creates based on the time_col column additional time-related variables, e.g. day or year and adds them to the
        df class attribute.

        Returns:
            None
        '''

        # Add year, day, month variables
        self.df["year"] = self.df[self.time_col].dt.year
        self.df["month"] = self.df[self.time_col].dt.month
        self.df["day"] = self.df[self.time_col].dt.day
        self.df["date"] = self.df[self.time_col].dt.date
        self.df["weekday_name"] = self.df[self.time_col].dt.day_name().astype("string") # Weekday name, e.g. Monday
        self.df["weekday"] = self.df[self.time_col].dt.weekday

    def add_daily_ridership(self):
        '''
        Calculates daily ridership per station and station / fare class and adds columns to df class attribute. Requires
        add_time_variables to have been run before that.

        Returns:
            None
        '''
        # Add daily ridership numbers per station
        self.df["daily_ridership"] = self.df\
            .groupby(["date", self.station_id_col])["ridership"]\
            .transform("sum")

        # Add daily ridership numbers per station and fare class
        self.df["daily_ridership_fare_class"] = self.df\
            .groupby(["date", self.fare_class_col, self.station_id_col])["ridership"]\
            .transform("sum")

    def add_daily_ridership_quantiles(self,
                                      quantiles: tuple[float] = (0.0, 0.25, 0.5, 0.75, 1.0),
                                      label_quantiles: list[str] = ["(0, 25]", "(25, 50]", "(50, 75]", "(75, 100]"]):
        '''
        Creates three new columns in the df class attribute called mean_daily_ridership, quant_group, and quant_val
        that contain the average daily ridership per station, a category for each station based on their location within
        the distribution of average daily riderships across the whole data set, and the corresponding bin used to create
        the category label.
        Args:
            quantiles (tuple[float], optional): Quantiles of the daily ridership columns to use for categorization
            label_quantiles (list[str], optional): List of labels to use for the bins, i.e. the categories
        Returns:
            None
        '''
        # Info print
        print(f"Creating grouping based on the quantiles: {quantiles}")

        ## Calculate average daily ridership per station
        # Subset to station / daily ridership values (to sped up computation)
        # Calculate average as a reduced df, as removing duplicates makes it not possible to add via transfom
        mean_df = self.df\
            .drop_duplicates(subset=[self.station_id_col, "date"])\
            .groupby(self.station_id_col)\
            .agg(mean_daily_ridership=("daily_ridership", "mean"))\
            .reset_index()
        self.df = pd.merge(self.df, mean_df, on=self.station_id_col, how="left")

        # Get quantile values to be able to apply pandas cut function
        ridership_quantiles_df = self.df\
                                     .drop_duplicates(subset=[self.station_id_col, "date"])\
                                     .loc[:, "mean_daily_ridership"]\
            .quantile(q=quantiles) # Has to include the left edge for binning

        # Add category labels and a column that contains the bins
        self.df["quant_group"] = pd.cut(self.df["mean_daily_ridership"].values,
                                        bins=ridership_quantiles_df.values,
                                        labels=label_quantiles, right=True,
                                        include_lowest=True)  # Lowest interval includes left edge
        self.df["quant_val"] = pd.cut(self.df["mean_daily_ridership"].values,
                                      bins=ridership_quantiles_df.values,
                                      right=True, include_lowest=True)  # Lowest interval includes left edge

    def model_data_prep(self,
                        station_ids: Union[list, pd.Series],
                        start_date: datetime.date,
                        end_date: datetime.date,
                        add_zeros: bool = True,
                        add_vars: list[str]=None,
                        add_vars_agg: list[str] = None,
                        remove_na: bool = False,
                        test_len: int = 14):
        '''
        Generates the training and test data to be used for forecasting for one period. Aggregates ridership per hour,
        per station and adds rows with ridership = 0 for hours with no passengers (can be overridden). Adds a column
        that indicates train / test split. Default returns only the time series, time_col, train / test indicator and
        station_id. Use add_vars and add_vars_agg to additionally get other variables.

        Args:
            station_ids (Union[list, pd.Series]): A list or pandas.Series of station ids to subset the data to
            start_date (datetime.date): A start date to use for the time horizon of the train and test data
            end_date (datetime.date): An end date to use for the time horizon of the train and test data
            add_zeros (bool, optional): Whether to add zeros for hours with no ridership, default is True
            add_vars (list[str], optional): A list of column names to also retrieve, requires add_vars_agg to be set.
            add_vars_agg (list[str], optional): A list of aggregate functions that are used to aggregate the
                corresponding column name in add_vars. Matching is done based on position. Examples include "mean" or
                "first".
            remove_na (bool, optional): Only relevant if add_vars and add_vars_agg have been specified. Whether NAs in
                additional columns created through zero ridership addition should be replaced by first non-null value
                per day.
            test_len (int, optional): The length of the test set in days. Default is two weeks. Will always use the last
                test_len days as test days.
        Returns:
            (pd.DataFrame): A data frame containing the training and test data, with at least four columns, one
                indicating the station id, one the time stamp, one for the train / test indicator and one the total
                ridership within that hour. Might include additional columns based on add_vars.
        '''
        # Check that if either add_vars or add_vars_agg is not None, the other is not None as well
        if (add_vars is not None) ^ (add_vars_agg is not None):
            raise ValueError("Either only add_vars or add_vars_agg specified, but both must be specified!")

        # Check that start_date occurs before end_date
        if end_date < start_date:
            raise ValueError(f"The start date occurs after end date! Got start: {start_date}, Got end: {end_date}")

        # Raise warning if test_len is longer than number of days between start and end
        if test_len >= (end_date-start_date).days:
            warnings.warn("Test length is at least as long as total time series.", UserWarning, stacklevel=2)

        # Check that station ids exist
        stations_exist = {*self.df[self.station_id_col].unique()}
        missing_stations = {*station_ids}.difference(stations_exist)
        if len(missing_stations) > 0:
            raise KeyError(f"Some stations are not in data! Missing stations: {missing_stations}")


        # Status adding zeros
        print("The option to add_zeros is set to True. Will add zero ridership rows to missing hours in data set.")

        ## Generate data set
        # Subset data first
        mask_stations = (self.df[self.station_id_col].isin(station_ids))
        mask_time = (self.df["date"] >= start_date) & (self.df["date"] <= end_date)
        if add_vars is not None:
            cols_to_take = [self.time_col, self.station_id_col, self.ridership_col, *add_vars]
        else:
            cols_to_take = [self.time_col, self.station_id_col, self.ridership_col]
        subset_df = self.df.loc[(mask_stations) & (mask_time), cols_to_take]

        # Aggregate hourly data per station across fare class
        if (add_vars is not None) & (add_vars_agg is not None):
            add_agg_dict = {k:v for k, v in zip(add_vars, add_vars_agg)} # Additional aggregations across additional vars
            add_agg_dict.update({"ridership": "sum"})
        else:
            add_agg_dict = {"ridership": "sum"}
        subset_df = subset_df.groupby([self.station_id_col, self.time_col]).agg(add_agg_dict).reset_index()

        # Add 0 ridership rows if hour does not exist depending on add_zeros, creates NA in additional columns
        # Station id should create groups where time_col is an unique index
        if add_zeros:
            subset_df = subset_df.groupby(self.station_id_col, group_keys=False)\
                .apply(self.fill_missing_hours, remove_na=remove_na)

        # Add train / test indicator column (test = 1)
        test_start = end_date - datetime.timedelta(days=test_len)
        subset_df["test_ind"] = np.where(subset_df[self.time_col].dt.date >= test_start, 1, 0)

        return subset_df
    def fill_missing_hours(self, group: pd.DataFrame,
                           remove_na: bool = False):
        '''
        Helper function to add rows to the data set in case of missing hourly data. Should be used in a groupby and
        apply combination, such that group corresponds to a pandas.DataFrame containing the data of one group.
        Requires that the grouping leads to a unique time index, when using time_col, i.e. time_col in group should
        uniquely identify each row.

        Ridership column is set to 0, other columns are set to NA.

        Args:
            group (pd.DataFrame): Pandas data frame subset such that time_col is a unique row index.
            remove_na (bool, optional): Only relevant if additional variables are present. Whether NAs in additional
            columns created through zero ridership addition should be replaced by first non-null value per day.
        Returns:
            (pandas.DataFrame) The group data frame but with added rows.
        '''
        # Check if time_col creates row index
        if group[self.time_col].nunique() != len(group):
            raise ValueError("Grouping does not return unique time index!")

        # Get current station id to fill station id nas
        station_id = group[self.station_id_col].unique()[0]

        # Set time_col as the index
        group.set_index(self.time_col, inplace=True)

        # Generate a date range for this group
        all_hours = pd.date_range(start=group.index.min(), end=group.index.max(), freq='H')

        # Reindex this group
        group = group.reindex(all_hours)

        # Fill missing "ridership" values with 0
        group['ridership'] = group['ridership'].fillna(0).astype(int)

        # Fill missing "station_id" values with the correct id
        group[self.station_id_col] = group[self.station_id_col].fillna(station_id).astype("string")

        # For each additional column, fill NaN values by the first non-NaN value of each day if specified
        if remove_na:
            additional_columns = [col for col in group.columns
                                  if col not in ['ridership', self.station_id_col, self.time_col]]
            for col in additional_columns:
                # First, fill NaN values forward for each additional column
                group[col] = group[col].ffill()

                # Then, ensure that each day starts with the first non-NaN value (if any) by grouping by day and
                # applying ffill again
                group[col] = group.groupby(group.index.date)[col].transform(lambda x: x.ffill())

        # Reset the index to have time_col as a separate column again
        group.reset_index(inplace=True)
        group.rename(columns={'index': self.time_col}, inplace=True)

        return group

if __name__ == "__main__":
    # Parameters
    dtypes_csv = {"transit_mode": "string",
                  "station_complex_id": "string",
                  "station_complex": "string",
                  "borough": "string",
                  "payment_method": "string",
                  "fare_class_category": "string",
                  "ridership": "Int64",
                  "transfers": "Int64",
                  "latitude": np.float32,
                  "longitude": np.float32,
                  "Counties": np.float32,
                  "NYS Municipal Boundaries": "Int32",
                  "New York Zip Codes": "Int32",
                  "Georeference": object}

    parse_dates = "transit_timestamp"
    # dir_data = os.path.join("..", "data",
    #                         "MTA_Subway_Hourly_Ridership__Beginning_February_2022_20240404_subset_long.csv")
    dir_data = os.path.join("..", "data",
                            "MTA_Subway_Hourly_Ridership__Beginning_February_2022_20240412.csv")

    # Instantiate class
    mta_data = DataLoaderRidership(dir_data,
                                   time_col="transit_timestamp",
                                   dtype_dict=dtypes_csv,
                                   date_format="%m/%d/%Y %I:%M:%S %p")
    # Add quartiles grouping
    mta_data.add_daily_ridership_quantiles()
    df = mta_data.df

    # test_station = df["station_complex_id"].unique()[:4]
    # start_date_test = datetime.date(2023, 2, 1)
    # end_date_test = start_date_test + datetime.timedelta(days=10)
    #
    # test_df = mta_data.model_data_prep(test_station, start_date_test, end_date_test,
    #                                    add_vars=["borough"], add_vars_agg=["first"],
    #                                    remove_na=True,
    #                                    test_len=3)
    # mask = test_df["ridership"] == 0
    # print(test_df[mask])
