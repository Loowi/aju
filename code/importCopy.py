import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import Sequence
import math
import random
import copy

# https: // sknadig.me/TensorFlow2.0-dataset/
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

# Create two datasets
data1 = [['tom', 10], ['nick', 15], ['juli', 14]]
data2 = [['tom2', 102], ['nick2', 152], ['juli2', 142]]
data3 = [['tom3', 1023], ['nick3', 1523], ['juli3', 1423]]
data4 = [['tom4', 10234], ['nick4', 15234], ['juli4', 14234]]
df1 = pd.DataFrame(data1, columns=['Name', 'Age'])
df2 = pd.DataFrame(data2, columns=['Name', 'Age'])
df3 = pd.DataFrame(data3, columns=['Name', 'Age'])
df4 = pd.DataFrame(data4, columns=['Name', 'Age'])

files = [df1, df2, df3, df4]


class testgen(Sequence):

    def __init__(self, files, num_files, batch_size):
        self.files = copy.deepcopy(files)
        self.num_files = num_files
        self.batch_size = batch_size

    def __len__(self):
        # Required to start a new epoch
        return 0

    def __getitem__(self, idx):
        listPandaFiles = [self.files.pop(0) for i in range(self.num_files)]
        df_new = pd.concat(listPandaFiles)
        print(df_new)
        print("Empty line")
        return df_new

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.files = self.random.shuffle(self.files)


random.shuffle(files)
ab = testgen(files, 2, 3)
# t1 = ab.__getitem__(0)
# t2 = ab.__getitem__(1)
# t3 = ab.__getitem__(0)


