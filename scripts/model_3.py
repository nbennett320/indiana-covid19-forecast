import os, logging
import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v2 import data
from tensorflow.keras import layers
from tensorflow.keras.layers.experimental import preprocessing
from util import print_separator
print(f'tf version: {tf.__version__}')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)

model_dir = "./train"
raw_dataset = pd.read_csv('./data/indiana_county_level_mobility_time_series_data_formatted_7.csv', index_col=0)
df = pd.DataFrame(raw_dataset)
# df['county'] = pd.Categorical(df['county'])
# df['state'] = pd.Categorical(df['state'])
# df['country'] = pd.Categorical(df['country'])
# df['combined_key'] = pd.Categorical(df['combined_key'])

dataset = raw_dataset.copy()
dataset.isna().sum()
dataset = df.dropna()

def model_cases():
  print(df)
  train_set = dataset.sample(frac=0.8, random_state=0)
  test_set = dataset.drop(train_set.index)
  train_set.describe().transpose()
  train_feats = train_set.copy()
  test_feats = test_set.copy()
  train_label = train_feats.pop('cases')
  test_label = test_feats.pop('cases')

  train_set.describe().transpose()[['mean', 'std']]
  normalizer = preprocessing.Normalization()
  normalizer.adapt(np.array(train_feats))
  print(normalizer.mean.numpy())

def define_categorical_vars():
  # create vocabulary
  # print(df)
  # print(df.loc[:, 'county'].values)
  covid_features = df.copy()
  covid_labels = covid_features.pop('cases')
  inputs = {}
  for name, column in covid_features.items():
    dtype = column.dtype
    if dtype == object:
      dtype = tf.string
    else:
      dtype = tf.float64
    inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
  print_separator()
  print(f'inputs: {inputs}')

  # get numeric inputs and normalize
  numeric_inputs = {
    name:input for name, input in inputs.items()
      if input.dtype == tf.float64
  }
  print_separator()
  print(f'numeric_inputs: {numeric_inputs}')
  x = layers.Concatenate()(list(numeric_inputs.values()))
  norm = preprocessing.Normalization()
  norm.adapt(np.array(df[numeric_inputs.keys()]))
  all_numeric_inputs = norm(x)
  preprocessed_inputs = [all_numeric_inputs]
  
  # append non-numeric inputs
  for name, col in covid_features.items():
    dtype = col.dtype
    if dtype != object:
      continue
    lookup = preprocessing.StringLookup(vocabulary=np.unique(covid_features[name]))
    one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())
    print(f'lookup {lookup.get_vocabulary()}')
    x = lookup(col)
    x = one_hot(x)
    preprocessed_inputs.append(x)
  print_separator()
  print(f'preprocessed inputs: {preprocessed_inputs}')

  # 
  preprocessed_inputs_cat = layers.Concatenate(axis=1)(preprocessed_inputs)
  covid_preprocessing_model = tf.keras.Model(inputs, preprocessed_inputs_cat)

  covid_features_dict = {
    name: np.array(value) for name, value in covid_features.items()
  }
  features_dict = {
    name: values[:1] for name, values in covid_features_dict.items()
  }

  print_separator()
  print(f'features dict: {features_dict}')
  optimizer = keras.optimizers.Adam(
    learning_rate=0.001,
    beta_1=0.9,
    beta_2=0.999
  )
  covid_preprocessing_model.compile(
    optimizer=optimizer, 
    loss='categorical_crossentropy',
    loss_weights=None
  )
  eval = covid_preprocessing_model.evaluate(
    x=features_dict,
    batch_size=32,
  )

  print(eval)

  # county_name_lookup = preprocessing.StringLookup()
  # county_name_lookup.adapt(df.loc)
  # county_name_embedding = layers.Embedding(
  #   input_dim=county_name_lookup.vocab_size(),
  #   output_dim=32
  # )
  # county_name_model = keras.Sequential([county_name_lookup, county_name_embedding])
  # print_separator()
  # print(f'county vocabulary: {county_name_lookup.get_vocabulary()[:3]}')
  # print(county_name_lookup(['porter', 'marion']))
  # # county_name_model(['porter', 'marion'])
  # # county_name_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])




  # state_name_lookup = preprocessing.StringLookup()
  # state_name_lookup.adapt(df.loc[:, 'state'].values)
  # state_name_embedding = layers.Embedding(
  #   input_dim=state_name_lookup.vocab_size(),
  #   output_dim=32
  # )
  # print_separator()
  # print(f'state vocabulary: {state_name_lookup.get_vocabulary()[:3]}')
  # print(state_name_lookup(['indiana']))

  # country_name_lookup = preprocessing.StringLookup()
  # country_name_lookup.adapt(df.loc[:, 'country'].values)
  # country_name_embedding = layers.Embedding(
  #   input_dim=country_name_lookup.vocab_size(),
  #   output_dim=32
  # )
  # print_separator()
  # print(f'country vocabulary: {country_name_lookup.get_vocabulary()[:3]}')
  # print(country_name_lookup(['US']))

  # combined_name_lookup = preprocessing.StringLookup()
  # combined_name_lookup.adapt(df.loc[:, 'combined_key'].values)
  # combined_name_embedding = layers.Embedding(
  #   input_dim=combined_name_lookup.vocab_size(),
  #   output_dim=32
  # )
  # print_separator()
  # print(f'combined vocabulary: {combined_name_lookup.get_vocabulary()[:3]}')
  # print(combined_name_lookup(['US']))


def main():
  define_categorical_vars()

main()