import numpy as np
import tensorflow as tf
from tensorflow import keras
from pathlib import WindowsPath
import tensorConvertNew
import chess


inputPath = WindowsPath('C:/Users/Watson/Projects/remus/data/')
model = keras.models.load_model(str(inputPath / 'watsonBrain'))


# fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/4K3 b KQkq - 0 1'
# fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
# fen = '4k3/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'

# fen = 'rnbqkbnr/ppp1pppp/8/3p4/8/8/PPPPPPPP/4K3 b KQkq - 0 1'

fen = 'rnbqkbnr/ppp1pppp/8/3p4/8/8/8/4K3 b KQkq - 0 1'



a = tensorConvertNew.fenToTensor(fen)

inputState = np.expand_dims(a, axis=0)
k = model.predict(inputState)
print(k)
board = chess.Board(fen)
board
