import glob
import pandas as pd
import numpy as np
# import tensorflow as tf
import anxilliaryfunctions as af
from tqdm import tqdm

# Input paths
files = glob.glob(
    'C:\\Users\\Watson\\Projects\\remus\\input\\final_input\\*.pkl')

# Convert and merge all file in the numpy arrays
stateList = []
moveList = []
resultList = []

for file in files:
    data = pd.read_pickle(file)

    # Extract and save states
    tqdm.pandas(mininterval=10)

    data['TensorState'] = data['Fen'].progress_apply(
        lambda x: af.fenToTensor(x))
    inputStates = data['TensorState'].values
    inputStates = np.array([x for x in inputStates])

    # Extract and save result and best move for a game
    target1 = data['Result'].values.astype(np.int8)

    target2 = af.convertMoves(data[['Move']])
    target2 = np.array([x for x in target2])

    # Save results
    stateList = stateList.append(inputStates)
    moveList = moveList.append(target1)
    resultList = resultList.append(target2)

stateArray = np.array(stateList)
movesArray = np.array(moveList)
resultArray = np.array(resultList)
