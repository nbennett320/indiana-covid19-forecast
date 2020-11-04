import numpy as np
import pandas as pd
import re, numbers, decimal, math
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana_by_county_formatted.csv', index_col=1)
county_level_data = pd.read_csv('./data/indiana_county_level_time_series_data_formatted_trimmed_date.csv', index_col=0)
county_df = pd.DataFrame(county_level_data)
vehicle_df = pd.DataFrame(mobility_vehicle_data)

def main():
  print(county_df.index)
  for i, j in vehicle_df.iterrows():
    date_range_county_mask = county_df.loc[i, 'county'][1]
    date_range_location_mask = j['sub_region_2']
    match = True if date_range_county_mask in str(date_range_location_mask) else False
    for k, x in j.items():
      if k == 'sub_region_2':
        continue
      
      county_df.at[i, k] = j[k]
      # print('k', k)
      # print('x', x)
      # print('loc', date_range_county_mask)
      # print('j subregion', date_range_location_mask)
      # print('matches', match)
      # print('i', i)
      # print('j', j)
      # print('updated val:',county_df.at[i, k])
    print(i)
  print(county_df)
  county_df.to_csv(r'./data/indiana_county_level_mobility_time_series_data_formatted_4.csv')

main()