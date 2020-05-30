import chess
import chess.engine

def stockfish_evaluation(board, time_limit = 1):
    engine = chess.engine.SimpleEngine.popen_uci("stockfish_20011801_x64.exe")
    result = engine.analyse(board, chess.engine.Limit(time=time_limit))
    return result['score']

board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
result = stockfish_evaluation(board)
print(result)



import chess.pgn

pgn = open("ficsgamesdb_2018.pgn")

list=[]
while True:
	game=chess.pgn.read_game(pgn)
	if game == None:
		break
	list.append(game)

#Store games
import pickle
import chess
import chess.engine
import chess
import chess.svg
from IPython.display import SVG
# with open("games.pkl", "wb") as fp:   #Pickling
#     pickle.dump(list[:10], fp)

with open("games.pkl", "rb") as fp:   # Unpickling
    games = pickle.load(fp)

# Create games
engine = chess.engine.SimpleEngine.popen_uci("stockfish_20011801_x64.exe")

import numpy as np
import pandas as pd
df = pd.DataFrame (columns = ['Game','Move','Fen','Evaluation'])

for i in range(1000):
    pass

for gameNum, game in enumerate(games):
    board = game.board()
    for moveNum, move in enumerate(game.mainline_moves()):
        board.push(move) 
        result = engine.analyse(board, chess.engine.Limit(time=0.1))
        # score = result['score'].score(mate_score=1000)
        score = result['score']
        fen = board.fen()
        
        # Store the result
        df = df.append({'Game': gameNum, 'Move': moveNum, 'Fen': fen, 'Evaluation': score}, ignore_index=True)
        
# Mate(-0) < Mate(-1) < Cp(-50) < Cp(200) < Mate(4) < Mate(1) < MateGiven
df['Evaluation'] = df['Evaluation'].astype(str)
df['Evaluation'] = ['+1000' if '#+' in x else x for x in df['Evaluation']]
df['Evaluation'] = ['-1000' if '#-' in x else x for x in df['Evaluation']]
df['Evaluation'] = pd.to_numeric(df['Evaluation'])
        
        

from sklearn.preprocessing import MaxAbsScaler
df['Evaluation2'] = MaxAbsScaler().fit(df['Evaluation'])
df['Evaluation2'] = np.clip(df['Evaluation'], -500, 500)
df['Evaluation2'] = np.interp(df['Evaluation2'], (df['Evaluation2'].min(), df['Evaluation2'].max()), (-1, +1))






SVG(chess.svg.board(board=board,size=400))


import multiprocessing as mp
mp.cpu_count()








board
board.board_fen

import chess
import chess.svg
from IPython.display import SVG
SVG(chess.svg.board(board=board,size=200))







import numpy as np

def batchtotensor(inputbatch):
    
    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1,9))
    pieces_dict = {pieces_str[0]:1, pieces_str[1]:2, pieces_str[2]:3, pieces_str[3]:4 , pieces_str[4]:5, pieces_str[5]:6,
              pieces_str[6]:-1, pieces_str[7]:-2, pieces_str[8]:-3, pieces_str[9]:-4 , pieces_str[10]:-5, pieces_str[11]:-6}

    maxnum = len(inputbatch)
    boardtensor = np.zeros((maxnum, 8, 8,7))
    
    for num, inputstr in enumerate(inputbatch):
        inputliste = inputstr.split()
        rownr = 0
        colnr = 0
        for i, c in enumerate(inputliste[0]):
            if c in pieces:
                boardtensor[num, rownr, colnr, np.abs(pieces_dict[c])-1] = np.sign(pieces_dict[c])
                colnr = colnr + 1
            elif c == '/':  # new row
                rownr = rownr + 1
                colnr = 0
            elif int(c) in valid_spaces:
                colnr = colnr + int(c)
            else:
                raise ValueError("invalid fenstr at index: {} char: {}".format(i, c))
        
        if inputliste[1] == "w":
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = 1
        else:
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = -1
  
    return boardtensor

import timeit
print(timeit.timeit(batchtotensor(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", '5rk1/1q1nbpp1/p1n1p2p/1p1p4/3P1B2/2N1PN2/PP1Q1PPP/5RK1 w']), number=1000))

import cProfile

cProfile.run('batchtotensor(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", "5rk1/1q1nbpp1/p1n1p2p/1p1p4/3P1B2/2N1PN2/PP1Q1PPP/5RK1 w"])', sort='time')

%time batchtotensor(["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", '5rk1/1q1nbpp1/p1n1p2p/1p1p4/3P1B2/2N1PN2/PP1Q1PPP/5RK1 w'])



fen = '5rk1/1q1nbpp1/p1n1p2p/1p1p4/3P1B2/2N1PN2/PP1Q1PPP/5RK1, w'
z = ["rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w", '5rk1/1q1nbpp1/p1n1p2p/1p1p4/3P1B2/2N1PN2/PP1Q1PPP/5RK1 w']
kala = batchtotensor(z)

import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Reshape, SimpleRNN
from tensorflow.keras.models import Model

sequence_length = 1
model = Sequential()
model.add(Reshape((sequence_length,448,), input_shape=(sequence_length,8,8,7)))
model.add(SimpleRNN(400,return_sequences=True))
model.add(SimpleRNN(400))
model.add(Dense(448, activation='tanh'))
model.add(Reshape((8,8,7)))
print(model.summary())


tf.test.is_gpu_available()

def myGenerator(games_list):
    while 1:
        for i in range(len(games_list)): 
            X_batch = []
            y_batch = []
            for j in range(len(games_list[i])-sequence_length-1):
                X_batch.append(batchtotensor(games_list[i][j:j+sequence_length]))
                y_batch.append(batchtotensor(games_list[i][j+sequence_length:j+sequence_length+1]))
                
            X_batch = np.array(X_batch)
            y_batch = np.array(y_batch)
            y_batch = np.squeeze(y_batch, axis=1)

            yield (X_batch, y_batch)
            

my_generator = myGenerator()
model.compile(loss='mse', optimizer='adam')
history = model.fit_generator(my_generator, steps_per_epoch = gamesnr, epochs = 1, verbose=1,  workers=2)
model.save('model_chess_2.h5')            
            
            