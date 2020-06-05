import math
from skimage.transform import resize
from skimage.io import imread
import numpy as np
import tensorflow as tf
import pandas as pd
from tensorflow.keras.utils import Sequence

# https: // sknadig.me/TensorFlow2.0-dataset/
# https://www.tensorflow.org/api_docs/python/tf/keras/utils/Sequence

# Create two datasets
data1 = [['tom', 10], ['nick', 15], ['juli', 14]]
data2 = [['tom2', 102], ['nick2', 152], ['juli2', 142]]
data3 = [['tom3', 1023], ['nick3', 1523], ['juli3', 1423]]
df = pd.DataFrame(data1, columns=['Name', 'Age'])
df2 = pd.DataFrame(data2, columns=['Name', 'Age'])
df3 = pd.DataFrame(data3, columns=['Name', 'Age'])


# Here, `x_set` is list of path to the images
# and `y_set` are the associated classes.

class CIFAR10Sequence(Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) *
                         self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) *
                         self.batch_size]

        return np.array([
            resize(imread(file_name), (200, 200))
            for file_name in batch_x]), np.array(batch_y)


# Create custom generator
def our_generator():
    for i in range(1000):
        x = np.random.rand(28, 28)
        y = np.random.randint(1, 10, size=1)
        yield x, y


# Generator samples
dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))

dataset = tf.data.Dataset.from_generator(our_generator, (tf.float32, tf.int16))
dataset = dataset.batch(batch_size=10)
dataset = dataset.repeat(count=2)
dataset = dataset.shuffle(buffer_size=1000)
iterator = dataset.make_one_shot_iterator()

for batch, (x, y) in enumerate(dataset):
    pass
print("batch: ", batch)
print("Data shape: ", x.shape, y.shape)

# Success criteria: all data is consumed on every epoch, batch size is correct
pass


def data_generator(file_list, batch_size = 20):
    i = 0
    while True:
        if i*batch_size >= len(file_list):  # This loop is used to run the generator indefinitely.
            i = 0
            np.random.shuffle(file_list)
        else:
            file_chunk = file_list[i*batch_size:(i+1)*batch_size] 
            data = []
            labels = []
            label_classes = ["Fault_1", "Fault_2", "Fault_3", "Fault_4", "Fault_5"]
            for file in file_chunk:
                temp = pd.read_csv(open(file,'r')) # Change this line to read any other type of file
                data.append(temp.values.reshape(32,32,1)) # Convert column data to matrix like data with one channel
                pattern = "^" + eval("file[14:21]")      # Pattern extracted from file_name
                for j in range(len(label_classes)):
                    if re.match(pattern, label_classes[j]): # Pattern is matched against different label_classes
                        labels.append(j)  
            data = np.asarray(data).reshape(-1,32,32,1)
            labels = np.asarray(labels)
            yield data, labels
            i = i + 1
