import numpy as np
import pandas as pd
data_time_series_cases = pd.read_csv('./data/time_series_covid19_confirmed_US_indiana.csv')
data_time_series_deaths = pd.read_csv('./data/time_series_covid19_deaths_US_indiana.csv')
data_time_series_cases = pd.DataFrame(data_time_series_cases)
data_time_series_deaths = pd.DataFrame(data_time_series_deaths)

# get cases and deaths in indiana, export to csv
def get_indiana_confirmed():
  counties_df = data_time_series_cases['Admin2']
  state_df = data_time_series_cases['Province_State']
  country_df = data_time_series_cases['Country_Region']
  combined_keys_df = data_time_series_cases['Combined_Key']
  fips_df = data_time_series_cases['FIPS']
  lats_df = data_time_series_cases['Lat']
  longs_df = data_time_series_cases['Long_']
  cases_df = data_time_series_cases.iloc[:,11:]
  deaths_df = data_time_series_deaths.iloc[:,12:]
  result_df = pd.DataFrame(columns=[
    'date', 
    'county', 
    'state', 
    'country', 
    'combined_key', 
    'lat', 
    'long', 
    'fips', 
    'cases', 
    'deaths'
  ])

  # populate data frame rows
  k = 0
  for i in cases_df.columns:
    for j in range(0, len(counties_df)):
      date = i
      new_row = {
        'date': date, 
        'county': counties_df[j], 
        'state': state_df[j], 
        'country': country_df[j], 
        'combined_key': combined_keys_df[j], 
        'lat': lats_df[j], 
        'long': longs_df[j],
        'fips': fips_df[j], 
        'cases': cases_df[i][j], 
        'deaths': deaths_df[i][j]
      }
      result_df.loc[k] = new_row
      k += 1
  print('result')
  print(result_df)
  result_df.to_csv(r'./data/indiana_county_level_time_series_data.csv')

# main
def main():
  get_indiana_confirmed()

main()