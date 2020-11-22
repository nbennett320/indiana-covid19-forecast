import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model_dir = "./train"
dataset = pd.read_csv('./data/indiana_county_level_mobility_time_series_data_formatted_7.csv', index_col=0)
df = pd.DataFrame(dataset)

def build_model():
  print(df.shape)
  print(df.columns)
  model = Sequential()
  model.add(Dense(
    units=512,
    input_shape=df.shape,
  ))
  model.compile(
    loss='mean_squared_error',
    optimizer=tf.keras.optimizers.Adam(1)
  )
  history = model.fit(
    df.index,
    df.cases,
    epochs=100,
  )
  print('Prediction: {}'.format(model.predict([10])))








#   feature_cols = [tf.feature_column.numeric_column(k) for k in df.columns]
#   print(feature_cols)
#   estimator = tf.estimator.LinearRegressor(
#     feature_columns=feature_cols,
#     model_dir=model_dir
#   )
#   estimator.train(input_fn=input_fn(
#       dataset,
#       num_epochs=None,
#       n_batch=128,
#       shuffle=False,
#     ),
#     steps=1000
#   )

# def input_fn(data, num_epochs=None, n_batch=128, shuffle=True):
#   return tf.compat.v1.estimator.inputs.pandas_input_fn(
#     x=pd.DataFrame({k: data[k].values for k in df.columns}),
#     y=pd.Series(df.index),
#     batch_size=n_batch,
#     num_epochs=num_epochs,
#     shuffle=shuffle
#   )

def main():
  build_model()

main()