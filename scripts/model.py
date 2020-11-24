from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections, sys, os, math, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pandas import tseries
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow_estimator import estimator as tfest
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from sklearn.linear_model import LinearRegression
from sklearn import model_selection
from scipy import interpolate
from datetime import datetime
from argparse import ArgumentParser
from functools import reduce
from util import print_separator, update_dataset, assign_school_type, format_date
tf = tf.compat.v2
tf.enable_v2_behavior()
tfb = tfp.bijectors
tfd = tfp.distributions

print(f'tf version: {tf.__version__}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
try:
  physical_devices = tf.config.list_physical_devices('GPU')
  tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
  print("no dedicated gpu or cannot modify virtual devices once initialized")
  pass

# default fp type
DTYPE = np.float32

# model train dir
global model_dir
model_dir = "./train/"

# county csv output dir
global output_dir
output_dir = False

# dataset dir
global dataset_dir
dataset_dir = './data/updated/'

# verbose mode
global is_verbose
is_verbose = False

# num of days to model for
global n_days
n_days = 14

global model_county
model_county = 'Porter'

# update dataset
global should_fetch_datasets
should_fetch_datasets = False

# plot results
global should_plot
should_plot = False

global show_smooth
show_smooth = False

# data filenames
indiana_counties_list_filename = './data/indiana_counties.csv'
county_level_combined_filename = './data/indiana_county_level_mobility_time_series_data_formatted_7.csv'
county_populations_by_age_filename = './data/indiana_population_by_age_by_county.csv'
prevention_measures_filename = './data/indiana_reopening_stages_prevention_measures.csv'
indiana_county_level_data_filename = './data/indiana_county_level_data.csv'
apple_vehicle_mobility_report_filename = './data/updated/data_apple_vehicle_mobility_report.csv'
google_mobility_trends_filename = './data/updated/data_google_regional_mobility_report/2020_US_Region_Mobility_Report.csv'
indiana_county_level_test_case_death_trends_filename = './data/updated/data_indiana_county_wide_test_case_death_trends.xlsx'
indiana_covid_demographics_by_county_filename = './data/updated/data_indiana_covid_demographics_by_county_and_district.xlsx'
indiana_hospital_vent_data_filename = './data/updated/data_indiana_hospital_vent_data.xlsx'
indiana_covid_cases_by_school_filename = './data/updated/data_indiana_covid_cases_by_school.xlsx'
jh_cases_filename = './data/updated/data_jh_cases.csv'
jh_deaths_filename = './data/updated/data_jh_deaths.csv'

# read static data
indiana_counties_raw = pd.read_csv(indiana_counties_list_filename)
indiana_county_level_data_raw = pd.read_csv(indiana_county_level_data_filename, index_col=0)
county_populations_by_age_raw = pd.read_csv(county_populations_by_age_filename, index_col=0)
prevention_measures_raw = pd.read_csv(prevention_measures_filename, index_col=0)
county_level_combined_raw = pd.read_csv(county_level_combined_filename, index_col=0)

# read fetched data
apple_vehicle_mobility_report_raw = pd.read_csv(apple_vehicle_mobility_report_filename, index_col=4)
google_mobility_trends_raw = pd.read_csv(google_mobility_trends_filename).set_index('sub_region_1')
indiana_county_level_test_case_death_trends_raw = pd.read_excel(indiana_county_level_test_case_death_trends_filename)
indiana_covid_demographics_by_county_raw = pd.read_excel(indiana_covid_demographics_by_county_filename)
indiana_hospital_vent_data_raw = pd.read_excel(indiana_hospital_vent_data_filename)
indiana_covid_cases_by_school_raw = pd.read_excel(indiana_covid_cases_by_school_filename)
jh_cases_raw = pd.read_csv(jh_cases_filename)

def format_apple_mobility_data():
  df = pd.DataFrame(apple_vehicle_mobility_report_raw).copy()
  cdf = pd.DataFrame(indiana_counties_raw).copy()
  df = df.loc['Indiana',:]
  df = df[df.geo_type != 'city']
  del df['geo_type']
  del df['alternative_name']
  del df['country']
  df.reset_index(drop=True, inplace=True)
  df['region'] = df['region'].apply(lambda x: x.replace('County', '').strip(' '))
  df.index = df['region']
  del df['region']
  del cdf['location_id']
  df = cdf.set_index('county_name').join(df)
  df = df.fillna(100)
  df = df.T
  df.loc['transportation_type',:] = df.loc['transportation_type',:].apply(lambda x: 'driving' if x == 100 else x)
  placeholder_df = pd.DataFrame(columns=['county_name', 'driving', 'walking', 'transit'])
  for col in df.columns:
    temp_df = pd.DataFrame(df.loc[:,col])
    temp_df.columns = temp_df.loc['transportation_type', :]
    temp_df.drop('transportation_type', inplace=True)
    temp_df.insert(0, 'county_name', col)
    placeholder_df = placeholder_df.append(temp_df)
  df = placeholder_df.fillna(100)
  df.index = pd.to_datetime(df.index)
  return df

def format_google_mobility_data():
  df = pd.DataFrame(google_mobility_trends_raw).copy()
  df = df.loc['Indiana', :]
  del df['country_region_code']
  del df['country_region']
  del df['metro_area']
  del df['iso_3166_2_code']
  del df['census_fips_code']
  df = df[df.sub_region_2 == df.sub_region_2]
  df['sub_region_2'] = df['sub_region_2'].apply(lambda x: x.replace('County', '').strip(' '))
  df.reset_index(drop=True, inplace=True)
  df.index = df['sub_region_2']
  df.reset_index(drop=True, inplace=True)
  df.index = df['date']
  del df['date']
  for col in df.columns[1:]:
    df[col] = df[col].apply(lambda x: 100 if math.isnan(DTYPE(x)) else DTYPE(x) + 100)
  df = df.rename(columns={ 'sub_region_2': 'county_name' })
  df = df.fillna(0)
  df.index = pd.to_datetime(df.index)
  return df

def format_county_level_test_case_death_trends():
  df = pd.DataFrame(indiana_county_level_test_case_death_trends_raw).copy()
  cdf = pd.DataFrame(indiana_counties_raw).copy()
  del cdf['location_id']
  for col in df.columns:
    df[col.lower()] = df[col]
    del df[col]
  df.set_index('county_name', inplace=True)
  # cumulative cases
  df.insert(1, 'covid_count_cumulative', 0)
  for i, j in cdf['county_name'].iteritems():
    df.loc[j, 'covid_count_cumulative'] = df.loc[j, 'covid_count'].cumsum()
    
  # cumulative deaths
  df.insert(2, 'covid_deaths_cumulative', 0)
  for i, j in cdf['county_name'].iteritems():
    df.loc[j, 'covid_deaths_cumulative'] = df.loc[j, 'covid_deaths'].cumsum()
  
  # # case covariance
  # df.insert(3, 'covid_count_covariance', 0)
  # for i, j in cdf['county_name'].iteritems():
  #   df.loc[j, 'covid_count_cumulative'] = pd.DataFrame([df.loc[j,:]['covid_count'].values.tolist(), df.loc[j,:]['date'].values.tolist()], columns=['covid_count', 'date']).cov()

  # # death covariance
  # df.insert(4, 'covid_deaths_covariance', 0)
  # for i, j in cdf['county_name'].iteritems():
  #   df.loc[j, 'covid_deaths_cumulative'] = df.loc[j, :].cov()

  df.sort_values(by=['date', 'county_name'], inplace=True)
  df.reset_index(inplace=True)
  df_date = df.pop('date')
  df_date = pd.to_datetime(df_date)
  df.insert(0, df_date.name, df_date)
  df_date = df_date.unique()
  # print('df date', df_date)
  # df.set_index('date', drop=False, inplace=True)
  del df['location_id']

  print('start', df)
  print('len:', len(df_date))
  for i in range(0, len(df_date)):
    if i % 20 == 0:
      print('[' + ('=' * int(math.ceil(i / 20.0))) + ']')
    sel = df.loc[df['date'] == df_date[i], :]
    isum = sel.sum()
    isum['county_name'] = 'Indiana'
    isum['date'] = df_date[i]
    isum = isum.to_frame().T
    df = df.append(isum, ignore_index=True)
  df.sort_values(by=['date', 'county_name'], inplace=True)
  print(df)
  print(df.loc[df['county_name'] == 'Indiana'])
  df.insert(5, 'covid_count_state_cumulative', df['covid_count'].cumsum())
  top = df.pop('county_name')
  df.insert(0, top.name, top)
  df = df.fillna(0)
  df.set_index('date', drop=True, inplace=True)
  df.sort_values(by=['county_name'], inplace=True)
  df.sort_index(axis=0, inplace=True)
  return df

def format_covid_demographics_by_county():
  df = pd.DataFrame(indiana_covid_demographics_by_county_raw).copy()
  # map suppressed data entries to -1
  for col in df.columns[1:]:
    df[col] = df[col].apply(lambda x: -1 if type(x) == str and str(x) in "Suppressed" else x)
  df = df.set_index('county_name')
  df = df[df.location_level != 'd']
  del df['location_level']
  del df['location_id']
  df = df.fillna(0)
  return df

def format_hospital_vent_data():
  df = pd.DataFrame(indiana_hospital_vent_data_raw).copy()
  jdf = pd.DataFrame(jh_cases_raw).copy()
  jdf = jdf.drop(labels=['UID', 'iso2', 'iso3', 'code3', 'FIPS', 'Combined_Key', 'Country_Region', 'Lat', 'Long_'], axis=1)
  jdf = jdf.loc[jdf['Province_State'] == 'Indiana', :]
  jdf = jdf.loc[jdf['Admin2'] != 'Out of IN', :]
  jdf = jdf.loc[jdf['Admin2'] != 'Unassigned', :]
  jdf = jdf.drop(labels=['Province_State', 'Admin2'], axis=1)
  jdf = jdf.reset_index()
  jdf = jdf.T
  jdf = jdf.drop(labels=['index'], axis=0)
  jdf = jdf.reset_index(drop=False)
  jdf.iloc[:, 0] = jdf.iloc[:, 0].apply(lambda x: format_date(x))
  jdf = jdf.set_index('index')
  jdf['cases_sum'] = jdf.sum(axis=1)
  for col in df.columns:
    df[col.lower()] = df[col]
    del df[col]
  df = df.set_index('date')
  df = df.join(jdf['cases_sum'])
  df = df.fillna(0)
  df.index = pd.to_datetime(df.index)
  return df

def format_covid_cases_by_school_data():
  df = pd.DataFrame(indiana_covid_cases_by_school_raw).copy()
  cdf = pd.DataFrame(indiana_counties_raw).copy()
  df['school_name'] = df['school_name'].apply(lambda x: x.lower())
  df['school_county'] = df['school_county'].apply(lambda x: x.capitalize())
  df['school_city'] = df['school_city'].apply(lambda x: x.lower() if x == x else -1)
  df['school_type_encoded'] = df['school_name'].apply(lambda x: assign_school_type(x))
  df.loc[:,'school_city_encoded'] = pd.factorize(df['school_city'])[0].reshape(-1,1)
  df.loc[:,'submission_status_encoded'] = pd.factorize(df['submission_status'])[0].reshape(-1,1)
  # del df['school_name']
  del df['school_city']
  del df['submission_status']
  del df['county_fips']
  del df['school_id']
  df['student_total_cases'] = df['student_total_cases'].apply(lambda x: np.random.randint(low=1, high=4) if x == '<5' else x)
  df['teacher_total_cases'] = df['teacher_total_cases'].apply(lambda x: np.random.randint(low=1, high=4) if x == '<5' else x)
  df['staff_total_cases'] = df['staff_total_cases'].apply(lambda x: np.random.randint(low=1, high=4) if x == '<5' else x)
  df['county_name'] = df['school_county']
  del df['school_county']
  del cdf['location_id']
  df = df.set_index('county_name').join(cdf)
  df = df.fillna(-1)
  df = df.rename(columns={ 'longitude': 'school_longitude', 'latitude': 'school_latitude' })
  del df['county_name']
  return df

def join_time_series_data(
  apple_mobility_df,
  google_mobility_df,
  county_level_test_case_death_trends_df,
  hospital_vent_df
):
  df_list = [
    apple_mobility_df,
    google_mobility_df,
    county_level_test_case_death_trends_df,
  ]
  for df in df_list:
    df.insert(0, 'date', df.index)
    df.set_index('date', inplace=True)
  df = pd.concat(df_list, axis=0)
  df.reset_index(inplace=True)
  df.set_index('date', inplace=True)
  df = df.fillna(0)
  hospital_vent_df.insert(0, 'date', hospital_vent_df.index)
  hospital_vent_df.set_index('date', inplace=True)
  df = df.join(hospital_vent_df)
  df = df.fillna(0)
  print(df)
  print(df.columns)
  print(df.covid_count)
  return df

def join_county_data(
  init,
  covid_demographics_by_county_df,
  covid_cases_by_school_df
):
  init = init.join(covid_demographics_by_county_df, rsuffix='_demo_by_county')
  print(init)
  init.insert(0, 'date', init.index)
  init.reset_index(inplace=True)
  init.set_index('county_name', inplace=True)
  df_list = [
    init,
    covid_cases_by_school_df
  ]
  for df in df_list:
    df.insert(0, 'county_name', df.index)
    df.set_index('county_name', inplace=True)
  df = pd.concat(df_list, axis=0)
  df.reset_index(inplace=True)
  df.set_index('date', inplace=True)
  print(df)
  sub_df = df.loc[np.datetime64('NaT')]
  sub_df.reset_index(inplace=True)
  sub_df.set_index('county_name', inplace=True)
  print(sub_df.columns)
  # df.reset_index(inplace=True)
  # df = df.set_index('county_name')
  # df = df.join(covid_demographics_by_county_df, rsuffix='_demo_by_county')
  # df = df.join(covid_cases_by_school_df)
  # print(df)
  # print(covid_demographics_by_county_df)

def join_static_data(init):
  df = pd.DataFrame(init).copy()
  print(df)

def preprocess_data():
  apple_mobility_df = format_apple_mobility_data()
  google_mobility_df = format_google_mobility_data()
  county_level_test_case_death_trends_df = format_county_level_test_case_death_trends()
  covid_demographics_by_county_df = format_covid_demographics_by_county()
  hospital_vent_df = format_hospital_vent_data()
  covid_cases_by_school_df = format_covid_cases_by_school_data()
  # time_series_df = join_time_series_data(
  #   apple_mobility_df,
  #   google_mobility_df,
  #   county_level_test_case_death_trends_df,
  #   hospital_vent_df
  # )

  if is_verbose:
    print("apple mobility:\n", apple_mobility_df)
    print("google mobility:\n", google_mobility_df)
    print("county_level_test_case_death_trends_df:\n", county_level_test_case_death_trends_df)
    print("county_level_test_case_death_trends_df:\n", covid_demographics_by_county_df)
    print("hospital_vent_df:\n", hospital_vent_df)
    print("covid_cases_by_school_df:\n", covid_cases_by_school_df)
    # print("all time series data:\n", time_series_df)

  # if model_county.lower() == 'all':
  #   cdf = pd.DataFrame(indiana_counties_raw).copy()
  #   del cdf['location_id']
  #   for i in cdf['county_name']:
  #     predict_covid_count(
  #       county_level_test_case_death_trends_df, 
  #       county=i
  #     )
  # else:
  #   predict_covid_count(
  #     county_level_test_case_death_trends_df, 
  #     county=model_county
  #   )
  predict_hospital_occupation(hospital_vent_df)

def predict_hospital_occupation(df: pd.DataFrame):
  fcols = []
  for col in df.columns:
    fcols.append(tf.feature_column.numeric_column(col))
  estimator = tf.estimator.LinearRegressor(
    feature_columns=fcols,
    model_dir=model_dir + '/hospital_occupation/',
    optimizer=tf.optimizers.Ftrl(
      learning_rate=0.025,
      l1_regularization_strength=0.0,
    )
  )
  if should_plot:
    plt.plot(
      df.index,
      df['beds_available_icu_beds_total'],
      label="bed data"
    )
  dft = df.copy()
  dft.reset_index(drop=True, inplace=True)
  train_x, test_x, train_y, test_y = model_selection.train_test_split(dft, dft['beds_available_icu_beds_total'])
  if is_verbose:
    print('train_x:', train_x)
    print('train_y:', train_y)
    print('test_x:', test_x)
    print('test_y:', test_y)
  train_fn = tfest.inputs.pandas_input_fn(
    x=train_x,
    y=train_y,
    shuffle=True,
    num_epochs=100,
    batch_size=14
  )
  test_fn = tfest.inputs.pandas_input_fn(
    x=test_x,
    y=test_y,
    shuffle=False,
    batch_size=14
  )
  model = estimator.train(input_fn=train_fn, steps=5000)
  result = model.evaluate(input_fn=test_fn, steps=10)
  for key, val in result.items():
    print(key, ':', val)
  print_separator()
  pred_generator = model.predict(input_fn=train_fn, yield_single_examples=False)
  predictions = None
  datelist = pd.date_range(datetime.today(), periods=n_days).to_numpy()
  for pred in pred_generator:
    for key, val in pred.items():
      predictions = val
      print(key, ':', val)
      print('len:', len(val))
      print('shape:', val.shape)
    break
  if should_plot:
    if is_verbose:
      print('predictions:',predictions.flatten())
    plt.plot(
      datelist,
      predictions.flatten(),
      label="bed predictions"
    )
    plt.show()
  if len(output_dir) > 0:
    pred_df = pd.DataFrame(data={'date': datelist, 'pred': predictions.flatten()})
    pred_df.set_index('date', inplace=True, drop=True)
    print(pred_df)
    polynomial_data = df.resample('5D', kind='timestamp').mean()
    polynomial_data = polynomial_data.resample('4H', kind='timestamp')
    polynomial_data = polynomial_data.interpolate(method='polynomial', order=3)
    polynomial_pred = pred_df.resample('2D', kind='timestamp').mean()
    polynomial_pred = polynomial_pred.resample('4H', kind='timestamp')
    polynomial_pred = polynomial_pred.interpolate(method='polynomial', order=3)
    output = dict({
      'x_data': df.index.values.tolist(),
      'y_data': dft['beds_available_icu_beds_total'].values.tolist(),
      'x_pred': datelist.tolist(),
      'y_pred': predictions.flatten().tolist(),
      'x_data_polynomial': polynomial_data.index.values.tolist(),
      'y_data_polynomial': polynomial_data.values.tolist(),
      'x_pred_polynomial': polynomial_pred.index.values.tolist(),
      'y_pred_polynomial': polynomial_data.values.tolist()
    })

    filename = output_dir + 'model_prediction_hospital_occupation.json'
    with open(filename, 'w') as outfile:
      json.dump(output, outfile)

def predict_covid_count(df: pd.DataFrame, county: str):
  df = df.loc[df['county_name'] == county, :]
  df.pop('county_name')
  fcols = []
  for col in df.columns:
    fcols.append(tf.feature_column.numeric_column(col))
  estimator = tf.estimator.LinearRegressor(
    feature_columns=fcols,
    model_dir=model_dir + '/covid_count/',
    optimizer=tf.optimizers.Ftrl(
      learning_rate=0.05,
      l1_regularization_strength=0.0,
    )
  )
  if should_plot:
    plt.plot(
      df.index,
      df['covid_count'],
      label="covid_count"
    )
  dft = df.copy()
  dft.reset_index(drop=True, inplace=True)
  train_x, test_x, train_y, test_y = model_selection.train_test_split(dft, dft['covid_count'])
  if is_verbose:
    print('train_x:', train_x)
    print('train_y:', train_y)
    print('test_x:', test_x)
    print('test_y:', test_y)
  train_fn = tfest.inputs.pandas_input_fn(
    x=train_x,
    y=train_y,
    shuffle=True,
    num_epochs=100,
    batch_size=14
  )
  test_fn = tfest.inputs.pandas_input_fn(
    x=test_x,
    y=test_y,
    shuffle=False,
    batch_size=14
  )
  model = estimator.train(input_fn=train_fn, steps=5000)
  result = model.evaluate(input_fn=test_fn, steps=10)
  for key, val in result.items():
    print(key, ':', val)
  print_separator()
  pred_generator = model.predict(input_fn=train_fn, yield_single_examples=False)
  predictions = None
  datelist = pd.date_range(datetime.today(), periods=n_days).to_numpy()
  for pred in pred_generator:
    for key, val in pred.items():
      predictions = val
      print(key, ':', val)
      print('len:', len(val))
      print('shape:', val.shape)
    break
  normalized_pred = [*map(lambda x: x + df['covid_count'].iat[-1], predictions.flatten())]

  # handle plotting
  if should_plot:
    if is_verbose:
      print('predictions:',predictions.flatten())
    plt.plot(
      datelist,
      normalized_pred,
      label="covid predictions"
    )
    plt.show()
  
  # handle exporting data
  if len(output_dir) > 0:
    if is_verbose:
      print('output', output_dir)

    # prepare prediction df
    df_pred = pd.DataFrame({
      'date': datelist,
      'covid_count': normalized_pred[len(normalized_pred)-n_days:]
    }).set_index('date')

    # calculate smoothed data
    polynomial_data = df['covid_count'].resample('5D', kind='timestamp').mean()
    polynomial_data = polynomial_data.resample('4H', kind='timestamp')
    polynomial_data = polynomial_data.interpolate(method='polynomial', order=3)
    polynomial_pred = df_pred['covid_count'].resample('2D', kind='timestamp').mean()
    polynomial_pred = polynomial_pred.resample('4H', kind='timestamp')
    polynomial_pred = polynomial_pred.interpolate(method='polynomial', order=3)
    output = dict({
      'county': county,
      'prediction_key': 'covid_count',
      'x_data': df['covid_count'].index.values.tolist(),
      'y_data': df['covid_count'].values.tolist(),
      'x_pred': datelist.tolist(),
      'y_pred': df_pred[len(df_pred)-n_days:].values.tolist(),
      'x_data_polynomial': polynomial_data.index.values.tolist(),
      'y_data_polynomial': polynomial_data.values.tolist(),
      'x_pred_polynomial': polynomial_pred.index.values.tolist(),
      'y_pred_polynomial': polynomial_data.values.tolist(),
    })

    filename = output_dir + 'model_prediction_' + county + '_' + 'covid_count' + '.json'
    with open(filename, 'w') as outfile:
      json.dump(output, outfile)

def plot_by_county(df, county='Marion', y=['covid_count', 'covid_deaths']):
  for n in range(0, len(y)):
    plot_line(
      df.loc[df['county_name'] == county, y[n]].index,
      df.loc[df['county_name'] == county, y[n]],
      legend_key=y[n].replace('_',' ')
    )
  title = "covid-19 in " + county.lower() + " county"
  format_plot(
    xlab="date",
    title=title,
    show_legend=True
  )
  plt.show()

def plot_line(time, series, format="-", start=0, end=None, legend_key=None):
  plt.plot(
    time[start:end], 
    series[start:end], 
    format,
    label=legend_key
  )

def format_plot(xlab=None, ylab=None, title=None, show_legend=True):
  plt.xlabel(xlab)
  plt.ylabel(ylab)
  if show_legend:
    plt.legend(loc='best')
  plt.title(title)

def predict_infections(
  intervention_indicators,
  population,
  initial_cases,
  mu,
  alpha_hier,
  conv_serial_interval,
  initial_days,
  total_days
):
  """Predict the number of infections by forward-simulation.

  Args:
    intervention_indicators: Binary array of shape
      `[num_countries, total_days, num_interventions]`, in which `1` indicates
      the intervention is active in that county at that time and `0` indicates
      otherwise.
    population: Vector of length `num_countries`. Population of each county.
    initial_cases: Array of shape `[batch_size, num_countries]`. Number of cases
      in each county at the start of the simulation.
    mu: Array of shape `[batch_size, num_countries]`. Initial reproduction rate
      (R_0) by county.
    alpha_hier: Array of shape `[batch_size, num_interventions]` representing
      the effectiveness of interventions.
    conv_serial_interval: Array of shape
      `[total_days - initial_days, total_days]` output from
      `make_conv_serial_interval`. Convolution kernel for serial interval
      distribution.
    initial_days: Integer, number of sequential days to seed infections after
      the 10th death in a county. (N0 in the authors' Stan code.)
    total_days: Integer, number of days of observed data plus days to forecast.
      (N2 in the authors' Stan code.)
  Returns:
    predicted_infections: Array of shape
      `[total_days, batch_size, num_countries]`. (Batched) predicted number of
      infections over time and by county.
  """
  alpha = alpha_hier - tf.cast(np.log(1.05) / 6.0, DTYPE)
  linear_prediction = tf.einsum('ijk,...k->j...i', intervention_indicators, alpha)
  rt = mu * tf.exp(-linear_prediction, name='reproduction_rate')
  daily_infections = tf.TensorArray(
      dtype=DTYPE, size=total_days, element_shape=initial_cases.shape)
  for i in range(initial_days):
    daily_infections = daily_infections.write(i, initial_cases)
  init_cumulative_infections = initial_cases * initial_days
  cond = lambda i, *_: i < total_days
  def body(i, prev_daily_infections, prev_cumulative_infections):
    # The probability distribution over days j that someone infected on day i
    # caught the virus from someone infected on day j.
    p_infected_on_day = tf.gather(conv_serial_interval, i - initial_days, axis=0)
    prev_daily_infections_array = prev_daily_infections.stack()
    to_sum = prev_daily_infections_array * mcmc_util.left_justified_expand_dims_like(
        p_infected_on_day, prev_daily_infections_array)
    convolution = tf.reduce_sum(to_sum, axis=0)
    rt_adj = ((population - prev_cumulative_infections) / population) * tf.gather(rt, i)
    new_infections = rt_adj * convolution
    # Update the prediction array and the cumulative number of infections.
    daily_infections = prev_daily_infections.write(i, new_infections)
    cumulative_infections = prev_cumulative_infections + new_infections
    return i + 1, daily_infections, cumulative_infections

  _, daily_infections_final, last_cumm_sum = tf.while_loop(
    cond, 
    body,
    (initial_days, daily_infections, init_cumulative_infections),
    maximum_iterations=(total_days - initial_days)
  )
  return daily_infections_final.stack()

def predict_deaths(predicted_infections, ifr_noise, conv_fatality_rate):
  """Expected number of reported deaths by county, by day.

  Args:
    predicted_infections: Array of shape
      `[total_days, batch_size, num_countries]` output from
      `predict_infections`.
    ifr_noise: Array of shape `[batch_size, num_countries]`. Noise in Infection
      Fatality Rate (IFR).
    conv_fatality_rate: Array of shape
      `[total_days - 1, total_days, num_countries]`. Convolutional kernel for
      calculating fatalities, output from `make_conv_fatality_rate`.
  Returns:
    predicted_deaths: Array of shape `[total_days, batch_size, num_countries]`.
      (Batched) predicted number of deaths over time and by county.
  """
  # Multiply the number of infections on day j by the probability of death
  # on day i given infection on day j, and sum over j. This yields the expected
  result_remainder = tf.einsum(
      'i...j,kij->k...j', predicted_infections, conv_fatality_rate) * ifr_noise

  # Concatenate the result with a vector of zeros so that the first day is
  # included.
  result_temp = 1e-15 * predicted_infections[:1]
  return tf.concat([result_temp, result_remainder], axis=0)

def make_jd_prior(num_countries, num_interventions):
  return tfd.JointDistributionSequentialAutoBatched([
    # Rate parameter for the distribution of initial cases (tau).
    tfd.Exponential(rate=tf.cast(0.03, DTYPE)),

    # Initial cases for each county.
    lambda tau: tfd.Sample(
    tfd.Exponential(rate=tf.cast(1, DTYPE) / tau),
      sample_shape=num_countries
    ),

    # Parameter in Negative Binomial model for deaths (psi).
    tfd.HalfNormal(scale=tf.cast(5, DTYPE)),

    # Parameter in the distribution over the initial reproduction number, R_0
    # (kappa).
    tfd.HalfNormal(scale=tf.cast(0.5, DTYPE)),

    # Initial reproduction number, R_0, for each county (mu).
    lambda kappa: tfd.Sample(
      tfd.TruncatedNormal(
        loc=3.28, 
        scale=kappa, 
        low=1e-5, 
        high=1e5
      ),
      sample_shape=num_countries
    ),

    # Impact of interventions (alpha; shared for all countries).
    tfd.Sample(
      tfd.Gamma(tf.cast(0.1667, DTYPE), 1), 
      sample_shape=num_interventions
    ),

    # Multiplicative noise in Infection Fatality Rate.
    tfd.Sample(
      tfd.TruncatedNormal(
        loc=tf.cast(1., DTYPE), 
        scale=0.1, low=1e-5, 
        high=1e5
      ),
      sample_shape=num_countries
    )
  ])

def make_likelihood_fn(
  intervention_indicators, 
  population, 
  deaths,
  infection_fatality_rate, 
  initial_days, 
  total_days
):
  # Create a mask for the initial days of simulated data, as they are not
  # counted in the likelihood.
  observed_deaths = tf.constant(deaths.T[np.newaxis, ...], dtype=DTYPE)
  mask_temp = deaths != -1
  mask_temp[:, :START_DAYS] = False
  observed_deaths_mask = tf.constant(mask_temp.T[np.newaxis, ...])

  conv_serial_interval = make_conv_serial_interval(initial_days, total_days)
  conv_fatality_rate = make_conv_fatality_rate(
    infection_fatality_rate, 
    total_days
  )

  def likelihood_fn(tau, initial_cases, psi, kappa, mu, alpha_hier, ifr_noise):
    # Run models for infections and expected deaths
    predicted_infections = predict_infections(
      intervention_indicators, 
      population, 
      initial_cases, 
      mu, 
      alpha_hier,
      conv_serial_interval, 
      initial_days, 
      total_days
    )
    e_deaths_all_countries = predict_deaths(
      predicted_infections, 
      ifr_noise, 
      conv_fatality_rate
    )

    # Construct the Negative Binomial distribution for deaths by county.
    mu_m = tf.transpose(e_deaths_all_countries, [1, 0, 2])
    psi_m = psi[..., tf.newaxis, tf.newaxis]
    probs = tf.clip_by_value(mu_m / (mu_m + psi_m), 1e-9, 1.)
    likelihood_elementwise = tfd.NegativeBinomial(
      total_count=psi_m, 
      probs=probs
    ).log_prob(observed_deaths)
    return tf.reduce_sum(
      tf.where(
          observed_deaths_mask,
          likelihood_elementwise,
          tf.zeros_like(likelihood_elementwise)
        ),
        axis=[-2, -1]
      )
  return likelihood_fn

def daily_fatality_probability(infection_fatality_rate, total_days):
  """Computes the probability of death `d` days after infection."""

  # Convert from alternative Gamma parametrization and construct distributions
  # for number of days from infection to onset and onset to death.
  concentration1 = tf.cast((1. / 0.86)**2, DTYPE)
  rate1 = concentration1 / 5.1
  concentration2 = tf.cast((1. / 0.45)**2, DTYPE)
  rate2 = concentration2 / 18.8
  infection_to_onset = tfd.Gamma(concentration=concentration1, rate=rate1)
  onset_to_death = tfd.Gamma(concentration=concentration2, rate=rate2)

  # Create empirical distribution for number of days from infection to death.
  inf_to_death_dist = tfd.Empirical(
      infection_to_onset.sample([5e6]) + onset_to_death.sample([5e6]))

  # Subtract the CDF value at day i from the value at day i + 1 to compute the
  # probability of death on day i given infection on day 0, and given that
  # death (not recovery) is the outcome.
  times = np.arange(total_days + 1., dtype=DTYPE) + 0.5
  cdf = inf_to_death_dist.cdf(times).numpy()
  f_before_ifr = cdf[1:] - cdf[:-1]
  # Explicitly set the zeroth value to the empirical cdf at time 1.5, to include
  # the mass between time 0 and time .5.
  f_before_ifr[0] = cdf[1]

  # Multiply the daily fatality rates conditional on infection and eventual
  # death (f_before_ifr) by the infection fatality rates (probability of death
  # given intection) to obtain the probability of death on day i conditional
  # on infection on day 0.
  return infection_fatality_rate[..., np.newaxis] * f_before_ifr

def make_conv_fatality_rate(infection_fatality_rate, total_days):
  """Computes the probability of death on day `i` given infection on day `j`."""
  p_fatal_all_countries = daily_fatality_probability(infection_fatality_rate, total_days)

  # Use the probability of death d days after infection in each county
  # to build an array of shape [total_days - 1, total_days, num_countries],
  # where the element [i, j, c] is the probability of death on day i+1 given
  # infection on day j in county c.
  conv_fatality_rate = np.zeros(
    [total_days - 1, 
    total_days, 
    p_fatal_all_countries.shape[0]]
  )
  for n in range(1, total_days):
    conv_fatality_rate[n - 1, 0:n, :] = (p_fatal_all_countries[:, n - 1::-1]).T
  return tf.constant(conv_fatality_rate, dtype=DTYPE)

def make_conv_serial_interval(initial_days, total_days):
  """Construct the convolutional kernel for infection timing."""

  g = tfd.Gamma(tf.cast(1. / (0.62**2), DTYPE), 1./(6.5*0.62**2))
  g_cdf = g.cdf(np.arange(total_days, dtype=DTYPE))

  # Approximate the probability mass function for the number of days between
  # successive infections.
  serial_interval = g_cdf[1:] - g_cdf[:-1]

  # `conv_serial_interval` is an array of shape
  # [total_days - initial_days, total_days] in which entry [i, j] contains the
  # probability that an individual infected on day i + initial_days caught the
  # virus from someone infected on day j.
  conv_serial_interval = np.zeros([total_days - initial_days, total_days])
  for n in range(initial_days, total_days):
    conv_serial_interval[n - initial_days, 0:n] = serial_interval[n - 1::-1]
  return tf.constant(conv_serial_interval, dtype=DTYPE) 

def get_flags():
  arg_parser = ArgumentParser()
  arg_parser.add_argument(
    '-d', 
    '--days', 
    type=int,
    dest='days',
    help="number of days to forecase, defaults to 14"
  )
  arg_parser.add_argument(
    '-C', 
    '--county', 
    type=str,
    dest='county',
    help="county to model data for"
  )
  arg_parser.add_argument(
    '-D', 
    '--train-dir', 
    type=str,
    dest='model_dir',
    help="directory for model files"
  )
  arg_parser.add_argument(
    '-o', 
    '--output-dir', 
    type=str,
    dest='output_dir',
    help="directory for output files"
  )
  arg_parser.add_argument(
    '-u', 
    '--update-datasets',
    action='store_true',
    dest='should_fetch_datasets',
    help="if passed, update datasets"
  )
  arg_parser.add_argument(
    '-v', 
    '--verbose',
    action='store_true',
    dest='verbose_mode',
    help="if passed, use verbose console messages"
  )
  arg_parser.add_argument(
    '-P', 
    '--plot',
    action='store_true',
    dest='plot_mode',
    help="if passed, plot results"
  )
  arg_parser.add_argument(
    '-S', 
    '--smooth-mode',
    type=str,
    dest='interpolation_method',
    help="pandas interpolation method for line smoothing, 'spline' or 'polynomial'"
  )
  args = arg_parser.parse_args()
  global n_days
  n_days = args.days if args.days else 14
  global model_county
  model_county = args.county if args.county else 'Porter'
  global model_dir
  model_dir = args.model_dir if args.model_dir else './train'
  global output_dir
  output_dir = args.output_dir if args.output_dir.lower() not in 'false' and args.output_dir.lower() not in 'none' else False
  global should_fetch_datasets
  should_fetch_datasets = args.should_fetch_datasets
  global should_plot
  should_plot = args.plot_mode
  global show_smooth
  show_smooth = args.interpolation_method if args.interpolation_method else False
  global is_verbose
  is_verbose = args.verbose_mode

def main():
  get_flags()
  if should_fetch_datasets:
    if is_verbose:
      print("updating datasets...")
    update_dataset(dir=dataset_dir)
    if is_verbose:
      print("done.")
    print_separator()
  preprocess_data()
  # predict_infections()

main()