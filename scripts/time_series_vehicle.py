import numpy as np
import pandas as pd
import re, numbers, decimal, math
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana_by_county_formatted_trimmed_date.csv', index_col=0)
county_level_data = pd.read_csv('./data/indiana_county_level_time_series_data_formatted_trimmed_date.csv', index_col=1)
county_df = pd.DataFrame(county_level_data)
vehicle_df = pd.DataFrame(mobility_vehicle_data)

vehicle_types = [
  'driving',
  'walking'
]

def main():
  for i, j in vehicle_df.iterrows():
    match = ''
    for k in county_df.index:
      print('k',k)
      if k in str(i):
        match = k
        break
      else:
        continue 
    print('match', match)
    
    county_mask = county_df.loc[k]
    dates = j[1:].index

    print('county_mask', county_mask)
    print(type(county_mask))
    print('i', i)
    print('j', j)
    print('dates', dates)

    for date in dates:
      print('j.date(value)', j[date], date)
      
      v_type = j['transportation_type']
      print(v_type)
      key = "mobility_by_" + v_type
      county_df.loc[county_df['date'] == date, key] = j[date]
      print('fin', county_mask.loc[county_mask['date'] == date, :])
  # print(county_df)
  # county_df.to_csv(r'./data/indiana_county_level_mobility_time_series_data_formatted_4.csv')

main()