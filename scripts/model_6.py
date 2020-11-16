from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections, os, math, logging
from argparse import ArgumentParser
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from util import print_separator
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
model_dir = "./train"

# num of epochs
global model_epochs
model_epochs = 100

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

def format_apple_mobility_data():
  df = pd.DataFrame(apple_vehicle_mobility_report_raw).copy()
  cdf = pd.DataFrame(indiana_counties_raw).copy()
  df = df.loc['Indiana', :]
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
  df = df.fillna(0)
  return df.T

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
  df = df.fillna(0)
  return df

def format_county_level_test_case_death_trends():
  df = pd.DataFrame(indiana_county_level_test_case_death_trends_raw).copy()
  for col in df.columns:
    df[col.lower()] = df[col]
    del df[col]
  df = df.set_index('date')
  del df['location_id']
  top = df.pop('county_name')
  df.insert(0, top.name, top)
  df = df.fillna(0)
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
  for col in df.columns:
    df[col.lower()] = df[col]
    del df[col]
  df = df.set_index('date')
  df = df.fillna(0)
  return df

def format_covid_cases_by_school_data():
  df = pd.DataFrame(indiana_covid_cases_by_school_raw).copy()
  cdf = pd.DataFrame(indiana_counties_raw).copy()
  df['school_name'] = df['school_name'].apply(lambda x: x.lower())
  df['school_county'] = df['school_county'].apply(lambda x: x.capitalize())
  df['school_city'] = df['school_city'].apply(lambda x: x.lower() if x == x else -1)
  dummies = pd.get_dummies(df['submission_status'])
  for col in dummies.columns:
    label = col.lower().replace(' ', '_').strip(' ')
    df[label] = dummies[col]
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

def preprocess_data():
  apple_mobility_df = format_apple_mobility_data()
  print("apple mobility:\n", apple_mobility_df)
  google_mobility_df = format_google_mobility_data()
  print("google mobility:\n", google_mobility_df)
  county_level_test_case_death_trends_df = format_county_level_test_case_death_trends()
  print("county_level_test_case_death_trends_df:\n", county_level_test_case_death_trends_df)
  covid_demographics_by_county_df = format_covid_demographics_by_county()
  print("county_level_test_case_death_trends_df:\n", covid_demographics_by_county_df)
  hospital_vent_df = format_hospital_vent_data()
  print("hospital_vent_df:\n", hospital_vent_df)
  covid_cases_by_school_df = format_covid_cases_by_school_data()
  print("covid_cases_by_school_df:\n", covid_cases_by_school_df)

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
      the intervention is active in that country at that time and `0` indicates
      otherwise.
    population: Vector of length `num_countries`. Population of each country.
    initial_cases: Array of shape `[batch_size, num_countries]`. Number of cases
      in each country at the start of the simulation.
    mu: Array of shape `[batch_size, num_countries]`. Initial reproduction rate
      (R_0) by country.
    alpha_hier: Array of shape `[batch_size, num_interventions]` representing
      the effectiveness of interventions.
    conv_serial_interval: Array of shape
      `[total_days - initial_days, total_days]` output from
      `make_conv_serial_interval`. Convolution kernel for serial interval
      distribution.
    initial_days: Integer, number of sequential days to seed infections after
      the 10th death in a country. (N0 in the authors' Stan code.)
    total_days: Integer, number of days of observed data plus days to forecast.
      (N2 in the authors' Stan code.)
  Returns:
    predicted_infections: Array of shape
      `[total_days, batch_size, num_countries]`. (Batched) predicted number of
      infections over time and by country.
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
  """Expected number of reported deaths by country, by day.

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
      (Batched) predicted number of deaths over time and by country.
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

    # Initial cases for each country.
    lambda tau: tfd.Sample(
    tfd.Exponential(rate=tf.cast(1, DTYPE) / tau),
      sample_shape=num_countries
    ),

    # Parameter in Negative Binomial model for deaths (psi).
    tfd.HalfNormal(scale=tf.cast(5, DTYPE)),

    # Parameter in the distribution over the initial reproduction number, R_0
    # (kappa).
    tfd.HalfNormal(scale=tf.cast(0.5, DTYPE)),

    # Initial reproduction number, R_0, for each country (mu).
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

    # Construct the Negative Binomial distribution for deaths by country.
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

  # Use the probability of death d days after infection in each country
  # to build an array of shape [total_days - 1, total_days, num_countries],
  # where the element [i, j, c] is the probability of death on day i+1 given
  # infection on day j in country c.
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
    '-e', 
    '--epochs', 
    type=int,
    dest='epochs',
    help="number of epochs"
  )
  arg_parser.add_argument(
    '-D', 
    '--train-dir', 
    type=str,
    dest='model_dir',
    help="directory for model files"
  )
  args = arg_parser.parse_args()
  global model_epochs
  model_epochs = args.epochs if args.epochs else 100
  global model_dir
  model_dir = args.model_dir if args.model_dir else './train'

def main():
  get_flags()
  preprocess_data()
  # predict_infections()

main()