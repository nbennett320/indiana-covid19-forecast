from numpy.testing._private.utils import rand
import pandas as pd
import numpy as np
import scipy
import tensorflow
import tensorflow_probability as tfp
tf = tensorflow.compat.v2
tf.enable_v2_behavior()
tfd = tfp.distributions

model_dir = "./train"
dataset = pd.read_csv('./data/indiana_county_level_mobility_time_series_data_formatted_7.csv', index_col=0)
df = pd.DataFrame(dataset)

def build_model():
  print(df.shape)
  print(df.columns)

def generate_data(n, d, link, scale=1, dtype=np.float32):
  model_coefficients = tfd.Uniform(
    low=-1,
    high=np.array(1, dtype)
  ).sample(
    d,
    seed=42
  )
  radius = np.sqrt(2.)
  model_coefficients *= radius / tf.linalg.norm(model_coefficients)
  mask = tf.random.shuffle(tf.range(d)) < int(0.5 * d)
  model_coefficients = tf.where(
      mask, model_coefficients, np.array(0., dtype))
  model_matrix = tfd.Normal(
      loc=0., scale=np.array(1, dtype)).sample([n, d], seed=43)
  scale = tf.convert_to_tensor(scale, dtype)
  linear_response = tf.linalg.matvec(model_matrix, model_coefficients)

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
  return model_matrix, response, model_coefficients, mask







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