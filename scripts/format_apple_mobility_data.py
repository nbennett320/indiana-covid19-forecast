import numpy as np
import pandas as pd
import re, numbers, decimal, math
from argparse import ArgumentParser



mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana_by_county_formatted_trimmed_date.csv', index_col=0)
county_level_data = pd.read_csv('./data/indiana_county_level_mobility_time_series_data_formatted_4.csv', index_col=0)
county_df = pd.DataFrame(county_level_data)
vehicle_df = pd.DataFrame(mobility_vehicle_data)

vehicle_types = [
  'driving',
  'walking',
  'transit'
]

def get_flags():
  arg_parser = ArgumentParser()
  arg_parser.add_argument(
    '-i', 
    '--input-file', 
    type=str,
    dest='input',
    help="input csv file"
  )
  arg_parser.add_argument(
    '-o', 
    '--output-file', 
    type=str,
    dest='output',
    help="output csv file"
  )
  args = arg_parser.parse_args()
  global model_epochs
  model_epochs = args.epochs if args.epochs else 100
  global model_dir
  model_dir = args.model_dir if args.model_dir else './train'

def main():
  for t in vehicle_types:
    key = "mobility_by_" + t
    county_df.insert(len(county_df.columns), key, 100)
  print(county_df)
  for i, j in vehicle_df.iterrows():
    print('=========================================================')
    match = ''
    for k in county_df['county']:
      # print('k',k)
      if k in str(i):
        match = k
        break
      else:
        continue 
    # print('match', match)
    
    # county_mask = county_df.loc[county_df[]]
    for x, y in j[1:].iteritems():
      print('---------------------------------------------')
      # print('i', i)
      # print('x', x)
      # print('y', y)
      v_type = j['transportation_type']
      key = "mobility_by_" + v_type
      # print('key', key)
      county_df.loc[(county_df.index == x) & (county_df.county == match), key] = y
      # print('sel', county_df.loc[(county_df.index == x) & (county_df.county == match), key])
      mask = county_df.loc[(county_df.index == x) & (county_df.county == match)]
      print('mask', mask)
  print(county_df)
  county_df.to_csv(r'./data/indiana_county_level_mobility_time_series_data_formatted_5.csv')

main()