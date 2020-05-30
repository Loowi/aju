import tensorflow as tf

dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])

for elem in dataset:
    print(elem.numpy())
