import pickle
import numpy as np
from tqdm import tqdm
import pandas as pd

# Import main file
infile = open('input_data_0.pkl', 'rb')
results = pickle.load(infile)
infile.close()


def batchtotensor(inputbatch):
    # if not isinstance(inputbatch, list):
    #     inputbatch = [inputbatch]

    pieces_str = "PNBRQK"
    pieces_str += pieces_str.lower()
    pieces = set(pieces_str)
    valid_spaces = set(range(1, 9))
    pieces_dict = {pieces_str[0]: 1, pieces_str[1]: 2, pieces_str[2]: 3, pieces_str[3]: 4, pieces_str[4]: 5, pieces_str[5]: 6,
                   pieces_str[6]: -1, pieces_str[7]: -2, pieces_str[8]: -3, pieces_str[9]: -4, pieces_str[10]: -5, pieces_str[11]: -6}

    maxnum = len(inputbatch)
    boardtensor = np.zeros((maxnum, 8, 8, 7))

    for num, inputstr in enumerate(inputbatch):
        inputliste = inputstr.split()
        rownr = 0
        colnr = 0
        for i, c in enumerate(inputliste[0]):
            if c in pieces:
                boardtensor[num, rownr, colnr, np.abs(
                    pieces_dict[c])-1] = np.sign(pieces_dict[c])
                colnr = colnr + 1
            elif c == '/':  # new row
                rownr = rownr + 1
                colnr = 0
            elif int(c) in valid_spaces:
                colnr = colnr + int(c)
            else:
                raise ValueError(
                    "invalid fenstr at index: {} char: {}".format(i, c))

        if inputliste[1] == "w":
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = 1
        else:
            for i in range(8):
                for j in range(8):
                    boardtensor[num, i, j, 6] = -1

    return boardtensor


tqdm.pandas(mininterval=10)
results['TensorState'] = results['Fen'].progress_apply(
    lambda x: batchtotensor([x]))

with open('allGamesTensor.pickle', 'wb') as f:
    pickle.dump(results, f)


# def parallelizeTasks(results, func, n_cores=6):
#     df_split = np.array_split(results, n_cores)
#     pool = Pool(n_cores)
#     df = pd.concat(pool.map(func, df_split))
#     pool.close()
#     pool.join()
#     return df

# def convertFen(df):
#     df['TensorState'] = df['Fen'].progress_apply(lambda x: batchtotensor([x]))
#     return df

# df2 = parallelizeTasks(results, convertFen, n_cores=6)

# fens = results['Fen'].tolist()


# from tqdm import tqdm


# tqdm.pandas()
# results['TensorState'] = results['Fen'].progress_apply(lambda x: batchtotensor(x))

# with open('allGamesTensor.pickle', 'wb') as f:
#         pickle.dump(results, f)

# # import timeit
# # print(timeit.timeit(batchtotensor(['rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1',
# #                                 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1']), number=1))
# # a = batchtotensor(['rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'])

# za = batchtotensor(['rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'])
