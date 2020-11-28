from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import collections, sys, os, math, json, logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow_estimator import estimator as tfest
from tensorflow_probability.python.internal import prefer_static as ps
from tensorflow_probability.python.mcmc.internal import util as mcmc_util
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn import model_selection
from scipy import interpolate
from datetime import datetime
from argparse import ArgumentParser
tf = tf.compat.v2
tf.enable_v2_behavior()
tfb = tfp.bijectors
tfd = tfp.distributions

# default fp type
DTYPE = np.float32

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