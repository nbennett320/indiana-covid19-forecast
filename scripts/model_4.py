from __future__ import print_function
import sys, os, logging
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
from argparse import ArgumentParser
from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.python.framework.tensor_conversion_registry import get
from tensorflow.python.keras.layers.preprocessing.normalization import Normalization
from tensorflow.python.ops.gen_array_ops import size
import tensorflow_probability as tfp
from util import print_separator
print(f'tf version: {tf.__version__}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
tfd = tfp.distributions

global model_epochs
model_epochs = 100
model_dir = "./train"
raw_dataset = pd.read_csv('./data/indiana_county_level_mobility_time_series_data_formatted_7.csv', parse_dates=['date'])
# df['county'] = pd.Categorical(df['county'])
# df['state'] = pd.Categorical(df['state'])
# df['country'] = pd.Categorical(df['country'])
# df['combined_key'] = pd.Categorical(df['combined_key'])

dataset = raw_dataset.copy()
# dataset.isna().sum()
# dataset = df.dropna()

def factorize_categorical_vars(df_target, cols):
  for col in cols:
    label = col + '_factorize_encode'
    df_target.loc[:, label] = pd.factorize(df_target[col])[0].reshape(-1, 1)
  print(df_target)
  print(df_target.columns)
  return df_target

def prep_frames(df_target):
  df_target = factorize_categorical_vars(df_target, ['date', 'county', 'state', 'country', 'combined_key'])
  df_target.drop('date', axis=1, inplace=True)
  df_target.drop('county', axis=1, inplace=True)
  df_target.drop('state', axis=1, inplace=True)
  df_target.drop('country', axis=1, inplace=True)
  df_target.drop('combined_key', axis=1, inplace=True)
  # df_target['date'] = df_target['date'].values.astype(np.int32)
  return df_target

def make_dataset(n, d, link, scale=1., dtype=np.float32):
  model_coefficients = tfd.Uniform(
      low=np.array(-1, dtype),
      high=np.array(1, dtype)).sample(d, seed=42)
  radius = np.sqrt(2.)
  model_coefficients *= radius / tf.linalg.norm(model_coefficients)
  model_matrix = tfd.Normal(
      loc=np.array(0, dtype),
      scale=np.array(1, dtype)).sample([n, d], seed=43)
  scale = tf.convert_to_tensor(scale, dtype)
  linear_response = tf.tensordot(
      model_matrix, model_coefficients, axes=[[1], [0]])
  if link == 'linear':
    response = tfd.Normal(loc=linear_response, scale=scale).sample(seed=44)
  elif link == 'probit':
    response = tf.cast(
        tfd.Normal(loc=linear_response, scale=scale).sample(seed=44) > 0,
        dtype)
  elif link == 'logit':
    response = tfd.Bernoulli(logits=linear_response).sample(seed=44)
  else:
    raise ValueError('unrecognized true link: {}'.format(link))
  return model_matrix, response, model_coefficients

def model():
  df = pd.DataFrame(raw_dataset)
  df = prep_frames(df)
  print(df)
  print(df.describe().transpose())
  train_set = df.sample(frac=0.8, random_state=0)
  test_set = df.drop(train_set.index)
  train_set.describe().transpose()
  train_feats = train_set.copy()
  test_feats = test_set.copy()
  train_label = train_feats.pop('cases')
  test_label = test_feats.pop('cases')

  print(f'train_set: {train_set}')
  print(f'train_feats: {train_feats}')
  print(f'train_label: {train_label}')

  train_set.describe().transpose()[['mean', 'std']]

  dates = np.array(train_feats['date_factorize_encode'])
  dates_normalizer = preprocessing.Normalization(input_shape=[1,])
  dates_normalizer.adapt(dates)
  dates_model = keras.Sequential([
    dates_normalizer,
    layers.Dense(units=1)
  ])
  
  dates_model.summary()
  pred = dates_model.predict(dates[:10])
  print(pred)

  optimizer = tf.optimizers.Adam(
    learning_rate=0.01,
    beta_1=0.9,
    beta_2=0.999
  )
  train_feats.sort_values(by='date_factorize_encode')
  # print("YO",train_feats['date_factorize_encode'])
  dates_model.compile(
    optimizer=optimizer,
    loss='mean_absolute_error',
  )
  history = dates_model.fit(
    train_feats['date_factorize_encode'],
    train_label,
    epochs=model_epochs,
    validation_split=0.2
  )
  hist = pd.DataFrame(history.history)
  hist['epoch'] = history.epoch
  tail = hist.tail()
  print(tail)
  # plot_loss(history)

  test_results = {}
  test_results['dates_model'] = dates_model.evaluate(
    test_feats['date_factorize_encode'],
    test_label
  )
  X = tf.linspace(
    0, 
    train_feats['date_factorize_encode'].values[-1], 
    train_feats['date_factorize_encode'].values[-1] - 4
  )
  print(type(X))
  X = tf.compat.v1.to_double(X)
  
  Y = dates_model.predict(dates[:140])
  Y = tf.compat.v1.to_double(Y)
  # pred = dates_model.predict(dates[:10])
  # X, Y, w_true = make_dataset(n=int(1e6), d=100, link='probit')
  # plot_cases(X, Y, train_feats, train_label)

  w, linear_response, is_converged, num_iter = tfp.glm.fit(
      model_matrix=X,
      response=X,
      model=tfp.glm.BernoulliNormalCDF())
  log_likelihood = tfp.glm.BernoulliNormalCDF().log_prob(Y, linear_response)

  print('is_converged: ', is_converged.numpy())
  print('    num_iter: ', num_iter.numpy())
  print('    accuracy: ', np.mean((linear_response > 0.) == tf.cast(Y, bool)))
  print('    deviance: ', 2. * np.mean(log_likelihood))
  print('||w0-w1||_2 / (1+||w0||_2): ', (np.linalg.norm(w_true - w, ord=2) /
                                        (1. + np.linalg.norm(w_true, ord=2))))
  print(w)


def plot_cases(x, y, train_feats, train_label):
  print(f'x: {x}')
  print(f'y: {y}')
  print(f'train feats: {train_feats}')
  print(f'type: {type(train_feats)}')
  print(f'train feats date: {train_feats["date_factorize_encode"]}')
  print(f'train label: {train_label}')
  plt.scatter(train_feats['date_factorize_encode'], train_label, label='cases')
  plt.plot(x, y, color='red', label='predictions')
  plt.xlabel('timestamp')
  plt.ylabel('cases')
  plt.legend()
  plt.show()

def get_flags():
  arg_parser = ArgumentParser()
  arg_parser.add_argument(
    '-e', 
    '--epochs', 
    type=int,
    dest='epochs',
    help="number of epochs"
  )
  args = arg_parser.parse_args()
  global model_epochs
  model_epochs = args.epochs if args.epochs else 100

def main():
  get_flags()
  model()

main()