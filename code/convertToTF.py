import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm
import modelChessTF
from pathlib import WindowsPath
import sys

# Import main file
inputPath = WindowsPath('C:/Users/Watson/Projects/remus/data/')

infile = open(inputPath / 'sampleGamesTensor.pickle', 'rb')
results = pickle.load(infile)
infile.close()

# Convert to tf data format and save to pickle
tqdm.pandas(mininterval=10)
inputState = results['TensorState'].progress_apply(
    lambda x: np.unpackbits(x, axis=-1))

states = tf.convert_to_tensor(inputState, dtype=tf.uint8)
target = results['Result'].values.astype(np.int8)

# Create TF data input file
dataset = tf.data.Dataset.from_tensor_slices((states, target))
with open(inputPath / 'tfInputData.pkl', 'wb') as f:
    pickle.dump([states, target], f)


# Create chess model
config = modelChessTF.ModelConfig()
model = modelChessTF.ChessModel(config)
model.build()

model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
_ = model.fit(dataset, batch_size=384, epochs=10,
              shuffle=True, validation_split=0.1)


# kala = inputState.values
# target = results['Result'].values.astype(np.int8)

# reet = np.asarray(kala)


# arg = tf.convert_to_tensor(inputState, dtype=tf.uint8)


# dataset = tf.data.Dataset.from_tensor_slices((arg))


# aq = inputState.values

# arg = tf.convert_to_tensor(kala, dtype=tf.float32)

# for x, y in dataset:
#     print(x, y)


# with open(inputPath / 'tfInputData.pkl', 'wb') as f:
#     pickle.dump(dataset, f)


# # Run code
# config = modelChessTF.ModelConfig()

# model = modelChessTF.ChessModel(config)
# model.build()
# model.model.summary()

# # Reserve 10,000 samples for validation
# dataset = dataset.shuffle()
# test_dataset = dataset.take(100000)
# train_dataset = dataset.skip(100000)

# # Run model
# model.compile(optimizer="Adam", loss="mse", metrics=["mae"])
# _ = model.fit(dataset, batch_size=384, epochs=10,
#               shuffle=True, validation_split=0.1)

# verbose = 0, validation_data = train_dataset)


# model.compile(optimizer = keras.optimizers.RMSprop(),  # Optimizer
#               # Loss function to minimize
#               loss = keras.losses.SparseCategoricalCrossentropy(
#                   from_logits=True),
#               # List of metrics to monitor
#               metrics = ['sparse_categorical_accuracy'])


# print('# Fit model on training data')
# history=model.fit(x_train, y_train,
#                     batch_size = 64,
#                     epochs = 3,
#                     # We pass some validation for
#                     # monitoring validation loss and metrics
#                     # at the end of each epoch
#                     validation_data = (x_val, y_val))

# print('\nhistory dict:', history.history)
