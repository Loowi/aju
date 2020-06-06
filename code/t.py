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

sasas

class testgen(Sequence):

    def __init__(self, files, num_files, batch_size):
        self.files = files
        self.num_files = num_files
        self.batch_size = batch_size
        self.files_enum = list(range(len(self.files)))

    def __len__(self):
        # Required to start a new epoch, not working now
        return 2

    def __getitem__(self, idx):
        if not self.files_enum:
            self.files_enum = list(range(len(self.files)))
            random.shuffle(self.files)

        filesNum = [self.files_enum.pop(0) for i in range(self.num_files)]
        listPandaFiles = [self.files[i] for i in filesNum]
        df_new = pd.concat(listPandaFiles)

        return df_new

    def on_epoch_end(self):
        # Updates indexes after each epoch, not working now
        self.files = random.shuffle(self.files)
        list(range(len(self.files)))


random.shuffle(files)
ab = testgen(files, 2, 3)
ab[1]
ab[1]
ab[1]
ab[1]
ab[1]
ab[1]
ab[1]
ab[1]
# t1 = ab.__getitem__(0)
# t2 = ab.__getitem__(1)
# t3 = ab.__getitem__(0)

