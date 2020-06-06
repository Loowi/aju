import numpy as np


def fenToTensor(input):
    # Define valid fen components
    pieces_white = {'P': 1, 'N': 2, 'B': 3, 'R': 4, 'Q': 5, 'K': 6,
                    'p': 7, 'n': 8, 'b': 9, 'r': 10, 'q': 11, 'k': 12}
    pieces_black = {'p': 1, 'n': 2, 'b': 3, 'r': 4, 'q': 5, 'k': 6,
                    'P': 7, 'N': 8, 'B': 9, 'R': 10, 'Q': 11, 'K': 12}

    # Split fen string into principal components
    splits = input.split()

    # Prepare tensor
    tensor = np.int8(np.zeros((12, 8, 8)))
    rownr = colnr = 0

    # If black has to move, turn the board
    if splits[1] == "b":
        pieces_white = pieces_black

    # Parse the string
    for i, c in enumerate(splits[0]):
        if c in pieces_white:
            tensor[pieces_white[c]-1, rownr, colnr] = 1
            colnr += 1
        elif c == '/':  # new row
            rownr += 1
            colnr = 0
        elif c.isdigit():
            colnr = colnr + int(c)
        else:
            raise ValueError("invalid fen string")

    # Pack into bits to save memory
    bitTensor = tensor
    # bitTensor = np.packbits(tensor, axis=-1)

    return bitTensor

# Test
# fen = 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1'
# a = fenToTensor(fen)
