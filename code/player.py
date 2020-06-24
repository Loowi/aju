"""
This encapsulates all of the functionality related to actually playing the game itself, not just
making / training predictions.
"""
from logging import getLogger
import chess
import numpy as np
import random
from pathlib import WindowsPath
from tensorflow import keras
import anxilliaryfunctions as af
from enum import Enum

logger = getLogger(__name__)


class Player:
    """
    Plays the actual game of chess, choosing moves based on policy and value network predictions
    """

    def __init__(self, model=None):
        self.model = model
        self.moves = []

    def move(self, game, enumDict):  # Create legitimate moves and pick one
        if self.model is None:
            moves = list(game.legal_moves)
            move = random.choice(moves)
            self.moves.append(move)
            value = np.nan

            return move, value

        else:
            gameState = game.fen()
            gameState = af.fenToTensor(gameState)

            inputState = np.expand_dims(gameState, axis=0)
            policy, value = self.model.predict(inputState)

            # Move with the highest value: find legit moves, max index and lookup in the dict
            moves = [str(x) for x in list(game.legal_moves)]
            kala = {move: policy[0][enumDict[move].value] for move in moves}
            move = max(kala, key=kala.get)
            self.moves.append(move)

            return move, value
