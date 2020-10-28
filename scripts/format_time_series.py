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
  # result_df = pd.DataFrame([
  #   counties_df,
  #   state_df,
  #   country_df,
  #   combined_keys_df,
  #   fips_df,
  #   lats_df,
  #   longs_df,
  # ]).T

  cases_df = data_time_series_cases.iloc[:,11:]
  deaths_df = data_time_series_deaths.iloc[:,12:]
  print("counties")
  print(counties_df)
  print("state")
  print(len(cases_df.columns))
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
  print(lats_df)
  print(cases_df[cases_df.columns[93]])
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
      # print('new row')
      # print(new_row)
      result_df.loc[k] = new_row
      k += 1
      # print(result_df)
      # print(k)
  print('result')
  print(result_df)



  # {
  #   'date', 
  #   'county', 
  #   'state', 
  #   'country', 
  #   'full_key', 
  #   'lat', 
  #   'long', 
  #   'fips', 
  #   'cases', 
  #   'deaths'
  # }
  # select cases and deaths per date
  # dates_df = data_time_series_cases.iloc[:,11:]
  # deaths_df = data_time_series_deaths.iloc[:,12:]

  # # append cases per day
  # for col in dates_df.columns:
  #   key = 'cases_' + col
  #   result_df[key] = dates_df[col]

  # # append deaths per day
  # for col in deaths_df.columns:
  #   key = 'deaths_' + col
  #   result_df[key] = deaths_df[col]
  result_df.to_csv(r'./data/indiana_county_level_time_series_data.csv')

# main
def main():
  get_indiana_confirmed()

main()