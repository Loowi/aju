from game import Game
from player import Player
import time
import anxilliaryfunctions as af
from tensorflow import keras
from pathlib import WindowsPath
from logging import getLogger


logger = getLogger(__name__)


inputPath = WindowsPath('C:/Users/Watson/Projects/remus/input/pkl_games/')
model = keras.models.load_model((inputPath / 'watsonBrainNew'))


t0 = time.time()
for i in range(1):
    white = Player(model)
    black = Player()
    game = Game(white, black)
    game.play_game()
t1 = time.time()

total_n = t1-t0
print("Cumulative time:", total_n)

game.board
