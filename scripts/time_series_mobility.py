import numpy as np
import pandas as pd
import re, numbers, decimal
county_level_data = pd.read_csv('./data/indiana_county_level_time_series_data_formatted_trimmed_date.csv', index_col=0)
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized_sorted.csv', index_col=0)
county_df = pd.DataFrame(county_level_data)
vehicle_df = pd.DataFrame(mobility_vehicle_data)
location_df = pd.DataFrame(mobility_location_data)

col_name = "retail_and_recreation_percent_change_from_baseline"

def main():
  county_df.insert(loc=30, column=col_name, value=100)
  # county_df['date'] = pd.to_datetime(county_df['date'])
  # location_df['date'] = pd.to_datetime(location_df['date'])
  # print(county_df)
  i = 0
  for date in location_df['date']:
    mask = county_df.loc[date]
    print("mask")
    print(mask.loc[date, col_name])
    # for county in mask['county']:
    #   print('ye')
    #   print(date)
    #   print(county)
    #   # print(location_df['sub_region_2'][i])
    #   try:
    #     if location_df['sub_region_2'][i].str.contains(county):
    #       county_df.at[date, 'county'] = location_df[col_name][i]
    #       print(county_df.at[date, 'county'])
    #   except:
    #     print("was float")
    #     print(i)
    i += 1







    # # print(i)
    # # date = county_df['date'][i]
    # print("date")
    # print(date)
    # # row = location_df.loc[location_df['date'].str.contains(date) & location_df['sub_region_2'].str.contains(county)]
    # # val = row[col_name]
    # # print('row:')
    # # print(row)
    # # row_loc = location_df.loc[location_df['date'] == date]
    # # print("row loc:")
    # # print(row_loc)
    # print(county_df['county'][i])
    # print(location_df['date'][i])

    # print(location_df['sub_region_2'][i])
    # print(location_df['sub_region_2'][i])
    # if location_df['date'][i] == date:
    #   print("match")
    #   print(i)
    # i += 1

    # county_df._set_value(i, col_name, location_mask)
  # print(county_df)
  # county_df.to_csv(r'./data/indiana_county_level_mobility_time_series_data_formatted_3.csv')

main()