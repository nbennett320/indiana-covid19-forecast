import numpy as np
import pandas as pd
import re
county_level_data = pd.read_csv('./data/indiana_county_level_time_series_data_formatted_2.csv')
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized.csv')
county_df = pd.DataFrame(county_level_data)
vehicle_df = pd.DataFrame(mobility_vehicle_data)
location_df = pd.DataFrame(mobility_location_data)

def main():
  county_df.insert(loc=33, column="retail_and_recreation_percent_change_from_baseline", value=100)
  print(county_df)
  i = 0
  for county in county_df['county']:
    date = county_df['date'][i]
    print(location_df['date'])
    print(i)
    # print(location_df['date'].loc[date])
    
    r_i = location_df['date'].at(date)
    print(r_i)
    print('rea')

    row_index = location_df['sub_region_2']
    print(row_index)
    l_row = location_df['date']
    print(l_row)

    # print(county)
    # print(date)
    # print(location_df['date'])
    j = 0
    for date_r in location_df['date']:
      if(date_r == date) & (location_df['sub_region_2'][j] == county):
        print(f'match: ({county}, {j})')
        county_df['retail_and_recreation_percent_change_from_baseline'][j] = location_df['retail_and_recreation_percent_change_from_baseline'][j]
    print(county_df['retail_and_recreation_percent_change_from_baseline'])
    row = location_df.loc[location_df['date'] in date]
    i += 1

main()