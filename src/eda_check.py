import pandas as pd
import numpy as np

dtypes_csv = {"transit_mode" : "str",
              "station_complex_id" : "str",
              "station_complex" : "str",
              "borough": "str",
              "payment_method" : "str",
              "fare_class_category" : "str",
              "ridership" : "Int64",
              "transfers" : "Int64",
              "latitude" : np.float64,
              "longitude" : np.float64,
              "Counties" : np.float64,
              "NYS Municipal Boundaries" : "Int64",
              "New York Zip Codes": "Int64",
              "Georeference" : object}

parse_dates = ["transit_timestamp"]

test = pd.read_csv("data/MTA_Subway_Hourly_Ridership__Beginning_February_2022_20240404_subset_long.csv",
                   dtype=dtypes_csv, parse_dates=parse_dates)

print(test.info())