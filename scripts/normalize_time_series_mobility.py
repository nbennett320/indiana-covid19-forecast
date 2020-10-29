import numpy as np
import pandas as pd
mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level.csv')
vehicle_df = pd.DataFrame(mobility_vehicle_data)
location_df = pd.DataFrame(mobility_location_data)

def main():
  print(location_df)
  df = location_df.apply(lambda x: x + 100 if x.name == 'retail_and_recreation_percent_change_from_baseline' else x)
  df = df.apply(lambda x: x + 100 if x.name == 'grocery_and_pharmacy_percent_change_from_baseline' else x)
  df = df.apply(lambda x: x + 100 if x.name == 'parks_percent_change_from_baseline' else x)
  df = df.apply(lambda x: x + 100 if x.name == 'workplaces_percent_change_from_baseline' else x)
  df = df.apply(lambda x: x + 100 if x.name == 'transit_stations_percent_change_from_baseline' else x)
  df = df.apply(lambda x: x + 100 if x.name == 'residential_percent_change_from_baseline' else x)
  print(df)
  df.to_csv(r'./data/2020_US_Region_Mobility_Report_indiana_county_level_normalized.csv')

main()