import numpy as np
import pandas as pd
import re
county_level_data = pd.read_csv('./data/indiana_county_level_time_series_data.csv')
# mobility_vehicle_data = pd.read_csv('./data/applemobilitytrends-2020-10-26-indiana.csv')
# mobility_location_data = pd.read_csv('./data/2020_US_Region_Mobility_Report_indiana_county_level.csv')
county_df = pd.DataFrame(county_level_data)
# vehicle_df = pd.DataFrame(mobility_vehicle_data)
# location_df = pd.DataFrame(mobility_location_data)

def main():
  print(county_df['date'])
  date_df = county_df['date'].apply(lambda x: format_date(x))
  county_df['date'] = date_df
  print(county_df)
  print(county_df['date'])
  county_df.to_csv(r'./data/indiana_county_level_time_series_data_formatted.csv')

def format_date(date):
  print(date)
  p = re.compile('([0-9]?[0-9])\/([0-9]?[0-9])\/([0-9][0-9])')
  m = p.match(date)
  month = '0' + m.group(1) if len(m.group(1)) < 2 else m.group(1)
  day =  '0' + m.group(2) if len(m.group(2)) < 2 else m.group(2)
  year = m.group(3)
  year = year + '20'
  f = year + '-' + month + '-' + day
  return f

def test():
  print(format_date('1/20/20'))
  print(format_date('12/20/20'))
  print(format_date('2/23/20'))
  print(format_date('10/23/20'))

main()
# test()