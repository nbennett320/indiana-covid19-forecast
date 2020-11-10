# prep model input dtypes
  # inputs = {}
  # for name, column in train_feats.items():
  #   dtype = column.dtype
  #   if dtype == object:
  #     dtype = tf.string
  #   else:
  #     dtype = tf.float64
  #   inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
  #   print(f'made input, {name}: {inputs[name]}')
  # print_separator()
  # print(f'inputs: {inputs}')

  # # filter all numeric inputs
  # numeric_inputs = {
  #   name:input for name, input in inputs.items()
  #     if input.dtype == tf.float64
  # }
  # print_separator()
  # print(f'numeric_inputs: {numeric_inputs}')
  # numeric_list = layers.Concatenate()(list(numeric_inputs.values()))
  # norm = preprocessing.Normalization()
  # norm.adapt(np.array(df[numeric_inputs.keys()]))
  # all_numeric_inputs = norm(numeric_list)
  # preprocessed_inputs = [all_numeric_inputs]

  # # append non-numeric inputs
  # for name, col in train_feats.items():
  #   dtype = col.dtype
  #   if dtype == object:
  #     lookup = preprocessing.StringLookup(vocabulary=np.unique(train_feats[name]))
  #     one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())
  #     print(f'lookup {lookup.get_vocabulary()}')
  #     x = lookup(col)
  #     x = one_hot(x)
  #     preprocessed_inputs.append(x)
  #   else:
  #     continue
  # print_separator()
  # print(f'preprocessed inputs: {preprocessed_inputs}')

  

  ################################### 
  # inputs = {}
  # for name, column in train_feats.items():
  #   dtype = column.dtype
  #   if dtype == object:
  #     dtype = tf.string
  #   else:
  #     dtype = tf.float64
  #   inputs[name] = tf.keras.Input(shape=(1,), name=name, dtype=dtype)
  # print_separator()
  # print(f'inputs: {inputs}')

  # # get numeric inputs and normalize
  # numeric_inputs = {
  #   name:input for name, input in inputs.items()
  #     if input.dtype == tf.float64
  # }
  # print_separator()
  # print(f'numeric_inputs: {numeric_inputs}')
  # x = layers.Concatenate()(list(numeric_inputs.values()))
  # norm = preprocessing.Normalization()
  # norm.adapt(np.array(df[numeric_inputs.keys()]))
  # all_numeric_inputs = norm(x)
  # preprocessed_inputs = [all_numeric_inputs]
  
  # # append non-numeric inputs
  # for name, col in train_feats.items():
  #   dtype = col.dtype
  #   if dtype == object:
  #     lookup = preprocessing.StringLookup(vocabulary=np.unique(train_feats[name]))
  #     one_hot = preprocessing.CategoryEncoding(max_tokens=lookup.vocab_size())
  #     print(f'lookup {lookup.get_vocabulary()}')
  #     x = lookup(col)
  #     x = one_hot(x)
  #     preprocessed_inputs.append(x)
  #   else:
  #     continue
  # print_separator()
  # print(f'preprocessed inputs: {preprocessed_inputs}')

  # preprocessed_inputs_cat = layers.Concatenate(axis=1)(preprocessed_inputs)
  # covid_preprocessing_model = tf.keras.Model(inputs, preprocessed_inputs_cat)

  # covid_features_dict = {
  #   name: np.array(value) for name, value in train_feats.items()
  # }
  # features_dict = {
  #   name: values[:1] for name, values in covid_features_dict.items()
  # }

  # print_separator()
  # print(f'features dict: {features_dict}')
  ############################################################

  # cases_normalizer = preprocessing.Normalization(input_shape=[1,])
  # cases_normalizer.adapt(cases)
  # cases_model = keras.Sequential([
  #   cases_normalizer,
  #   layers.Dense(units=1)
  # ])
  # cases_model.summary()
  # pred = cases_model.predict(train_feats['cases'])
  # print(pred)

  # normalizer = preprocessing.Normalization()
  # normalizer.adapt(np.array(train_feats))
  # print(normalizer.mean.numpy())

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



  # print(eval)

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