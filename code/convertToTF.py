import numpy as np
import pickle
import tensorflow as tf
from tqdm import tqdm


# Import main file
infile = open('sampleGamesTensor.pickle', 'rb')
results = pickle.load(infile)
infile.close()

# Convert to tf data format
tqdm.pandas(mininterval=10)
kala = results['TensorState'].progress_apply(
    lambda x: np.unpackbits(x, axis=-1))

ab = results
target = ab.pop('Result')
dataset = tf.data.Dataset.from_tensor_slices((kala, target.values))