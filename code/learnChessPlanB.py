from pathlib import WindowsPath
import pandas as pd
import numpy as np
import tensorflow as tf
import chessModel
# import chessModelSimple
from tensorflow.keras.mixed_precision import experimental as mixed_precision
from datetime import datetime
from collections import OrderedDict

# Input paths
inputPath = WindowsPath('C:/Users/Watson/Projects/remus/input/pkl_games')
data = pd.read_pickle(inputPath / 'ficsgamesdb_2010_updated_short.pkl')

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)

# Prepare data
inputStates = data['TensorState'].values
inputStates = np.array([x for x in inputStates])
states = tf.convert_to_tensor(inputStates, dtype=tf.int8)
target1 = data['Result'].values.astype(np.int8)
targetFinal1 = tf.convert_to_tensor(target1, dtype=tf.int8)

# Alternative paths
# inputPath = WindowsPath('C:/Users/Watson/Projects/remus/data/')
# with open(inputPath / 'tfInputData.pkl', 'rb') as f:
#     states, target = pickle.load(f)

# targetFinalScore = tf.convert_to_tensor(target, dtype=tf.uint8)


def convertMoves(moves):

    # Create function to convert one value
    def convert(move):
        # array2 = array1.copy()
        moveLabels = chessModel.create_uci_labels()
        labelDict = OrderedDict([(i, 0) for i in moveLabels])
        labelDict[move] = 1
        d = np.array(list(labelDict.values())).astype(np.float16)
        return d

    # Apply this to all cells
    values = moves.apply(lambda row: convert(row['Move']), axis=1)

    return values.values


target2 = convertMoves(data[['Move']])  # target2[200000].sum() = 1841
target2[200000].sum()

target2 = np.array([x for x in target2])
targetFinal2 = tf.convert_to_tensor(target2)


# # Create chess model
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
config = chessModel.ModelConfig()
model = chessModel.ChessModel(config)
model.build()

opt = tf.keras.optimizers.Adam()
losses = ['categorical_crossentropy', 'mean_squared_error']
model.compile(optimizer=opt, loss=losses, metrics=["mae"],
              loss_weights=[0.1, 0.9])

logs = inputPath / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logs),
                                                      histogram_freq=1,
                                                      profile_batch='100,110')

kala = model.fit(dataset=states, y=[targetFinal2, targetFinal1], batch_size=384,
                 epochs=3, shuffle=True, val_split=0.2, validation_data=None,
                 callbacks=[tensorboard_callback])

kala.save(str(inputPath / 'watsonBrainNew'))

# Create chess model
# policy = mixed_precision.Policy('mixed_float16')
# mixed_precision.set_policy(policy)
# config = chessModel.ModelConfig()
# model = chessModelSimple.ChessModelSimple(config)
# model.build()

# opt = tf.keras.optimizers.Adam()
# losses = ['mean_squared_error']
# model.compile(optimizer=opt, loss=losses, metrics=["mae"])

# logs = inputPath / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logs),
#                                                       histogram_freq=1,
#                                                       profile_batch='100,110')

# kala = model.fit(dataset=states, y=targetFinal1, batch_size=128, epochs=3,
#                  shuffle=True, val_split=0.2,
#                  validation_data=None,
#                  callbacks=[tensorboard_callback])
