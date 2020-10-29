import numpy as np
import pandas as pd
county_level_data = pd.read_csv('./data/indiana_county_level_time_series_data.csv')
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized.csv')
county_df = pd.DataFrame(county_level_data)
vehicle_df = pd.DataFrame(mobility_vehicle_data)
location_df = pd.DataFrame(mobility_location_data)

def main():
  print(county_df)

main()