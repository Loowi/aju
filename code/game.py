"""
Encapsulates the functionality for representing
and operating on the chess environment.
"""
import chess.pgn
import numpy as np
import copy
import chess
import anxilliaryfunctions as af

from logging import getLogger

logger = getLogger(__name__)

finalDict, enumDict = af.createMoveDict()


class Game:
    """
    Represents a chess environment where a chess game is played/
    Attributes:
        :ivar chess.Board board: current board state
        :ivar int num_halfmoves: number of half moves performed in total by each player
        :ivar Winner winner: winner of the game
        :ivar boolean resigned: whether non-winner resigned
        :ivar str result: str encoding of the result, 1-0, 0-1, or 1/2-1/2
    """

    def __init__(self, player1=None, player2=None):
        self.board = chess.Board()
        self.num_halfmoves = 0
        self.result = None
        self.white_to_move = 1
        self.finished = False
        self.Player1 = player1
        self.Player2 = player2
        self.enumDict = self.createMoveDict()

    def createMoveDict(self):
        finalDict, enumDict = af.createMoveDict()
        return enumDict

    def step(self, move):
        print(str(move))
        kala = chess.Move.from_uci(str(move))
        self.board.push(kala)
        self.num_halfmoves += 1
        self.white_to_move = 1 - self.white_to_move

    def game_over(self):
        self.finished = self.board.is_game_over()
        self.result = self.board.result()

    def display_pgn(self):
        pgn = chess.pgn.Game.from_board(self.board)
        print(pgn)
        # https://chesstempo.com/pgn-viewer/

    def play_game(self):
        print(1)
        while not self.finished:
            print(2)
            if self.white_to_move:
                print(3)
                move, value = self.Player1.move(self.board, enumDict)
                print('Board value:', value)
            else:
                print(4)
                move, value = self.Player2.move(self.board, enumDict)
                print('Board value:', value)

            print(5)
            self.step(move)
            print(6)
            self.game_over()
