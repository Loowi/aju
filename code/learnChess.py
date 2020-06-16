from pathlib import WindowsPath
import pandas as pd
import numpy as np
import tensorflow as tf
import chessModel
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from datetime import datetime
from collections import OrderedDict

# Input paths
inputPath = WindowsPath('C:/Users/Watson/Projects/remus/input/pkl_games')
data = pd.read_pickle(inputPath / 'ficsgamesdb_2010_updated_short.pkl')

# Prepare data
inputStates = data['TensorState'].values
inputStates = np.array([x for x in inputStates])
states = tf.convert_to_tensor(inputStates, dtype=tf.uint8)

target1 = data['Result'].values.astype(np.int8)


def convertMoves(moves):
    # Create labels
    moveLabels = chessModel.create_uci_labels()
    labelDict = OrderedDict([(i, 0) for i in moveLabels])

    # Create function to convert one value
    def convert(labelDict, move):
        labelDict[move] = 1
        d = np.array(list(labelDict.values())).astype(np.float16)
        return d

    # Apply this to all cells
    values = moves.apply(lambda row: convert(labelDict, row['Move']), axis=1)

    return values.values


target2 = convertMoves(data[['Move']])


# Create dataset
inputs = dict(states=states)
outputs = dict(outcomes1=target1, outcomes2=target1)
dataset = tf.data.Dataset.from_tensor_slices((states, [target1, target2]))


# dataset = tf.data.Dataset.from_tensor_slices((states, target1))
dataset = dataset.batch(384, drop_remainder=True)
test_dataset = dataset.take(200)
train_dataset = dataset.skip(200)
dataset = dataset.prefetch(50)
dataset = dataset.cache()

# Create chess model
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

config = chessModel.ModelConfig()
model = chessModel.ChessModel(config)
model.build()

opt = tf.keras.optimizers.Adam()

model.compile(optimizer=opt, loss="mse", metrics=["mae"])

logs = inputPath / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logs),
                                                      histogram_freq=1,
                                                      profile_batch='100,110')

kala = model.fit(dataset=train_dataset, batch_size=None, epochs=3,
                 shuffle=True, val_split=None,
                 validation_data=test_dataset,
                 callbacks=[tensorboard_callback])


# # train the network to perform multi-output classification
# H = model.fit(trainX,
# 	{"category_output": trainCategoryY, "color_output": trainColorY},
# 	validation_data=(testX,
# 		{"category_output": testCategoryY, "color_output": testColorY}),
# 	epochs=EPOCHS,
# 	verbose=1)
# # save the model to disk
# print("[INFO] serializing network...")
# model.save(args["model"])
