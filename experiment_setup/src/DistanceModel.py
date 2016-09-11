models = []

# Word vectors
model_word = Sequential()
model_word.add(Embedding(1e4, 300, input_length=1))
model_word.add(Reshape(dims=(300,)))
models.append(model_word)

# Context vectors
model_context = Sequential()
model_context.add(Embedding(1e4, 300, input_length=1))
model_context.add(Reshape(dims=(300,)))
models.append(model_context)

# Combined model
model = Sequential()
model.add(Merge(models, mode='dot'))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='mean_squared_error', optimizer=Adam(lr=0.001))