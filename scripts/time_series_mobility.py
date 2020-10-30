import numpy as np
import pandas as pd
import re, numbers, decimal
county_level_data = pd.read_csv('./data/indiana_county_level_time_series_data_formatted_2.csv')
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized.csv')
county_df = pd.DataFrame(county_level_data)
vehicle_df = pd.DataFrame(mobility_vehicle_data)
location_df = pd.DataFrame(mobility_location_data)

col_name = "retail_and_recreation_percent_change_from_baseline"

def main():
  county_df.insert(loc=31, column=col_name, value=100)
  print(county_df)
  i = 0
  for county in county_df['county']:
    print(i)
    date = county_df['date'][i]
    row = location_df.loc[location_df['date'].str.contains(date) & location_df['sub_region_2'].str.contains(county)]
    val = row[col_name]
    county_df[col_name] = val
    print(row)
    i += 1
  print(county_df)
  county_df.to_csv(r'./data/indiana_county_level_mobility_time_series_data_formatted.csv')

main()