import numpy as np
import pandas as pd
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level.csv', index_col=7)
vehicle_df = pd.DataFrame(mobility_vehicle_data)
location_df = pd.DataFrame(mobility_location_data)

cols = [
  'retail_and_recreation_percent_change_from_baseline',
  'grocery_and_pharmacy_percent_change_from_baseline',
  'parks_percent_change_from_baseline',
  'transit_stations_percent_change_from_baseline',
  'workplaces_percent_change_from_baseline',
  'residential_percent_change_from_baseline'
]

def main():
  print(location_df)
  for i, j in location_df.iterrows():
    for k in cols:
      j[k] += 100
    print('i', i)
    print('j', j)
  print(location_df)
  location_df.to_csv(r'./data/2020_US_Region_Mobility_Report_indiana_county_level_renormalized_2.csv')

main()