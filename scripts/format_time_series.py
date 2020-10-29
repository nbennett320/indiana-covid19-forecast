import numpy as np
import pandas as pd
data_time_series_cases = pd.read_csv('./data/time_series_covid19_confirmed_US_indiana.csv')
data_time_series_deaths = pd.read_csv('./data/time_series_covid19_deaths_US_indiana.csv')
county_level_data = pd.read_csv('./data/indiana_county_level_data.csv')
data_time_series_cases = pd.DataFrame(data_time_series_cases)
data_time_series_deaths = pd.DataFrame(data_time_series_deaths)
county_level_data_df = pd.DataFrame(county_level_data)
counties_df = data_time_series_cases['Admin2']
state_df = data_time_series_cases['Province_State']
country_df = data_time_series_cases['Country_Region']
combined_keys_df = data_time_series_cases['Combined_Key']
fips_df = data_time_series_cases['FIPS']
lats_df = data_time_series_cases['Lat']
longs_df = data_time_series_cases['Long_']

print(county_level_data_df['median_gross_rent'])

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
    'std_deaths_last_2_weeks',
    'median_gross_rent',
    'average_household_size',
    'building_permits_number',
    'percent_households_with_computer',
    'percent_households_with_broadband_internet',
    'dollars_per_capita_income_in_past_12_months_2018',
    'population_per_square_mile',
    'median_household_income',
    '2019_population_estimate',
    '2010_population',
    'percent_housing_units_in_multi_unit_structures',
    'percent_under_65_without_health_insurance',
    'percent_under_65_with_disability'
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
        'avg_cases_last_week': calc_n_week_average(j, k, 1, cases_df[i]),
        'avg_deaths_last_week': calc_n_week_average(j, k, 1, deaths_df[i]),
        'avg_cases_last_2_weeks': calc_n_week_average(j, k, 2, cases_df[i]),
        'avg_deaths_last_2_weeks': calc_n_week_average(j, k, 2, deaths_df[i]),
        'std_cases_last_week': calc_n_week_std(a, 1, cases_df[i]),
        'std_deaths_last_week': calc_n_week_std(a, 1, deaths_df[i]),
        'std_cases_last_2_weeks': calc_n_week_std(a, 2, cases_df[i]),
        'std_deaths_last_2_weeks': calc_n_week_std(a, 2, deaths_df[i]),
        'median_gross_rent': county_level_data_df['median_gross_rent'][j],
        'average_household_size': county_level_data_df['average_household_size'][j],
        'building_permits_number': county_level_data_df['building_permits_number'][j],
        'percent_households_with_computer': county_level_data_df['percent_households_with_computer'][j],
        'percent_households_with_broadband_internet': county_level_data_df['percent_households_with_broadband_internet'][j],
        'dollars_per_capita_income_in_past_12_months_2018': county_level_data_df['dollars_per_capita_income_in_past_12_months_2018'][j],
        'population_per_square_mile': county_level_data_df['population_per_square_mile'][j],
        'median_household_income': county_level_data_df['median_household_income'][j],
        '2019_population_estimate': county_level_data_df['2019_population_estimate'][j],
        '2010_population': county_level_data_df['2010_population'][j],
        'percent_housing_units_in_multi_unit_structures': county_level_data_df['percent_housing_units_in_multi_unit_structures'][j],
        'percent_under_65_without_health_insurance': county_level_data_df['percent_under_65_without_health_insurance'][j],
        'percent_under_65_with_disability': county_level_data_df['percent_under_65_with_disability'][j]
      }
      result_df.loc[k] = new_row
      k += 1
      a += 1
      print(new_row)
  print('result')
  print(result_df)
  result_df.to_csv(r'./data/indiana_county_level_time_series_data.csv')

def calc_n_week_average(index, row, weeks, df):
  n_sum = 0
  selection_df = df[index - 7 * weeks:index]
  for i in selection_df:
    n_sum += i
  print(f'avg: {float(n_sum / 7 * weeks)}')
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