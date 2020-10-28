import numpy as np
import pandas as pd
data_time_series_cases = pd.read_csv('./data/time_series_covid19_confirmed_US_indiana.csv')
data_time_series_deaths = pd.read_csv('./data/time_series_covid19_deaths_US_indiana.csv')
data_time_series_cases = pd.DataFrame(data_time_series_cases)
data_time_series_deaths = pd.DataFrame(data_time_series_deaths)
counties_df = data_time_series_cases['Admin2']
state_df = data_time_series_cases['Province_State']
country_df = data_time_series_cases['Country_Region']
combined_keys_df = data_time_series_cases['Combined_Key']
fips_df = data_time_series_cases['FIPS']
lats_df = data_time_series_cases['Lat']
longs_df = data_time_series_cases['Long_']

# get cases and deaths in indiana, export to csv
def get_indiana_confirmed():
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
    'deaths',
    'avg_cases_last_week',
    'avg_deaths_last_week',
    'avg_cases_last_2_weeks',
    'avg_deaths_last_2_week',
    'std_cases_last_week',
    'std_deaths_last_week',
    'std_cases_last_2_weeks',
    'std_deaths_last_2_weeks'
  ])

  # populate data frame rows
  k = 0
  print(len(cases_df.columns))
  for i in cases_df.columns:
    a = 0
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
        'deaths': deaths_df[i][j],
        'avg_cases_last_week': calc_n_week_average(a, 1, cases_df[i]),
        'avg_deaths_last_week': calc_n_week_average(a, 1, deaths_df[i]),
        'avg_cases_last_2_weeks': calc_n_week_average(a, 2, cases_df[i]),
        'avg_deaths_last_2_weeks': calc_n_week_average(a, 2, deaths_df[i]),
        'std_cases_last_week': calc_n_week_std(a, 1, cases_df[i]),
        'std_deaths_last_week': calc_n_week_std(a, 1, deaths_df[i]),
        'std_cases_last_2_weeks': calc_n_week_std(a, 2, cases_df[i]),
        'std_deaths_last_2_weeks': calc_n_week_std(a, 2, deaths_df[i])
      }
      result_df.loc[k] = new_row
      k += 1
      a += 1
      print(new_row)
  print('result')
  print(result_df)
  result_df.to_csv(r'./data/indiana_county_level_time_series_data.csv')

def calc_n_week_average(index, weeks, df):
  if(index - 1 < 7 * weeks):
    return 0
  else:
    n_sum = 0
    for i in range(index - 7 * weeks, index):
      n_sum += df[i]
    return float(n_sum / 7 * weeks)

def calc_n_week_std(index, weeks, df):
  if(index - 1 < 7 * weeks):
    return 0
  else:
    return np.std(df[index - 7 * weeks:index])

# main
def main():
  get_indiana_confirmed()

main()