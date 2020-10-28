import numpy as np
import pandas as pd
data_time_series_cases = pd.read_csv('./data/time_series_covid19_confirmed_US_indiana.csv')
data_time_series_deaths = pd.read_csv('./data/time_series_covid19_deaths_US_indiana.csv')
data_time_series_cases = pd.DataFrame(data_time_series_cases)
data_time_series_deaths = pd.DataFrame(data_time_series_deaths)

# 
def get_indiana_confirmed():
  counties_df = data_time_series_cases['Admin2']
  fips_df = data_time_series_cases['FIPS']
  lats_df = data_time_series_cases['Lat']
  longs_df = data_time_series_cases['Long_']
  result_df = pd.DataFrame([
    counties_df,
    fips_df,
    lats_df,
    longs_df,
  ]).T

  # select cases and deaths per date
  dates_df = data_time_series_cases.iloc[:,11:]
  deaths_df = data_time_series_deaths.iloc[:,12:]

  # append cases per day
  for col in dates_df.columns:
    key = 'cases_' + col
    result_df[key] = dates_df[col]

  # append deaths per day
  for col in deaths_df.columns:
    key = 'deaths_' + col
    result_df[key] = deaths_df[col]
  result_df.to_csv(r'./data/indiana_county_level_time_series_data.csv')

# main
def main():
  get_indiana_confirmed()

main()