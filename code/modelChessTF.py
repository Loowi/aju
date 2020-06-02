# Defines the actual model for making policy and value predictions given an observation.
from tensorflow.keras import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers.convolutional import Conv2D
from tensorflow.python.keras.layers.core import Activation, Dense, Flatten
from tensorflow.python.keras.layers.merge import Add
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.regularizers import l2
from tensorflow import keras


class ModelConfig:
    cnn_filter_num = 256
    cnn_first_filter_size = 5
    cnn_filter_size = 3
    res_layer_num = 7
    l2_reg = 1e-4
    value_fc_size = 256
    distributed = False
    input_depth = 18


class ChessModel:
    """
    The model which can be trained to take observations of a game of chess and return value and policy
    predictions.

    Attributes:
        :ivar Config config: configuration to use
        :ivar Model model: the Keras model to use for predictions
        :ivar digest: basically just a hash of the file containing the weights being used by this model
        :ivar ChessModelAPI api: the api to use to listen for and then return this models predictions (on a pipe).
    """

    def __init__(self, config):
        self.config = config
        self.model = None  # type: Model
        self.digest = None
        self.api = None

    def build(self):
        """
        Builds the full Keras model and stores it in self.model.
        """
        mc = self.config
        in_x = x = Input((12, 8, 8))

        # (batch, channels, height, width)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_first_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="input_conv-"+str(mc.cnn_first_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="input_batchnorm")(x)
        x = Activation("relu", name="input_relu")(x)

        for i in range(mc.res_layer_num):
            x = self._build_residual_block(x, i + 1)

        res_out = x

        # for value output
        x = Conv2D(filters=4, kernel_size=1, data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name="value_conv-1-4")(res_out)
        x = BatchNormalization(axis=1, name="value_batchnorm")(x)
        x = Activation("relu", name="value_relu")(x)
        x = Flatten(name="value_flatten")(x)
        x = Dense(mc.value_fc_size, kernel_regularizer=l2(
            mc.l2_reg), activation="relu", name="value_dense")(x)
        value_out = Dense(1, kernel_regularizer=l2(mc.l2_reg),
                          activation="tanh", name="value_out")(x)

        self.model = Model(in_x, [value_out], name="chess_model")

    def _build_residual_block(self, x, index):
        # mc = self.config.model
        mc = self.config

        in_x = x
        res_name = "res"+str(index)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name+"_conv1-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name=res_name+"_batchnorm1")(x)
        x = Activation("relu", name=res_name+"_relu1")(x)
        x = Conv2D(filters=mc.cnn_filter_num, kernel_size=mc.cnn_filter_size, padding="same",
                   data_format="channels_first", use_bias=False, kernel_regularizer=l2(mc.l2_reg),
                   name=res_name+"_conv2-"+str(mc.cnn_filter_size)+"-"+str(mc.cnn_filter_num))(x)
        x = BatchNormalization(axis=1, name="res"+str(index)+"_batchnorm2")(x)
        x = Add(name=res_name+"_add")([in_x, x])
        x = Activation("relu", name=res_name+"_relu2")(x)
        return x

    def compile(self, optimizer, loss, metrics):
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        return self.model

    def fit(self, dataset, batch_size, epochs, shuffle, validation_split, validation_data, callbacks):
        self.model.fit(x=dataset, batch_size=batch_size, epochs=epochs,
                       shuffle=True, validation_split=validation_split,
                       validation_data=validation_data,
                       callbacks=callbacks)
        return self.model

    def predict(self, x, batch_size=None, steps=None, callbacks=None, max_queue_size=10,
                workers=1, use_multiprocessing=False):
        value = self.model.predict(x=x, batch_size=batch_size, steps=steps,
                                   callbacks=callbacks, max_queue_size=max_queue_size, workers=workers,
                                   use_multiprocessing=use_multiprocessing)
        return value

# Config = ModelConfig()
# a = ChessModel(Config)
# a.build()
# a.model.summary()
# keras.utils.plot_model(
#     a.model, "my_first_model_with_shape_info.png", show_shapes=True)
