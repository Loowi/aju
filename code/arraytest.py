import pickle
import tensorflow as tf
import modelChessTF
from pathlib import WindowsPath
from datetime import datetime
from tensorflow.keras.mixed_precision import experimental as mixed_precision

# Import data
inputPath = WindowsPath('C:/Users/Watson/Projects/remus/data/')
with open(inputPath / 'tfInputData.pkl', 'rb') as f:
    states, target = pickle.load(f)

# Create TF data input file
dataset = tf.data.Dataset.from_tensor_slices((states, target))
dataset = dataset.batch(384, drop_remainder=True)

test_dataset = dataset.take(200)
train_dataset = dataset.skip(200)
dataset = dataset.prefetch(50)
dataset = dataset.cache()


# Create custome
dataset = tf.data.Dataset.from_generator(custom_generator).batch(batch_size)


# Create chess model
tf.keras.backend.set_floatx('float32')
print(tf.keras.backend.floatx())

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

config = modelChessTF.ModelConfig()
model = modelChessTF.ChessModel(config)
model.build()


opt = tf.keras.optimizers.Adam()

# opt = tf.keras.optimizers.SGD()

model.compile(optimizer=opt, loss="mse", metrics=["mae"])

logs = inputPath / "logs" / datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logs),
                                                      histogram_freq=1,
                                                      profile_batch='100,110')

# tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=str(logs))


kala = model.fit(dataset=train_dataset, batch_size=None, epochs=10,
                 shuffle=True, validation_split=None,
                 validation_data=test_dataset,
                 callbacks=[tensorboard_callback])

kala.save(str(inputPath / 'watsonBrain'))


# tensorboard --logdir="C:\Users\Watson\Projects\remus\data\logs"

# import numpy as np
# unique, counts = np.unique(target, return_counts=True)
# dict(zip(unique, counts))
