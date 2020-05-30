import pickle
import numpy as np
# from tqdm import tqdm
import pandas as pd
import tensorflow as tf

# Import main file
infile = open('allGames.pickle', 'rb')
results = pickle.load(infile)
infile.close()

df_split = np.array_split(results, 20)

for num, df in enumerate(df_split):
    fileName = ''.join(['input_data_', str(num), '.pkl'])
    with open(fileName, 'wb') as f:
        pickle.dump(df, f)



# import pickle

infile = open('input_data_0.pkl', 'rb')
results = pickle.load(infile)
infile.close()










target = results.pop('Result')
dataset = tf.data.Dataset.from_tensor_slices((results.values, target.values))
for feat, targ in dataset.take(1):
    print('Features: {}, Target: {}'.format(feat, targ))
