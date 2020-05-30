# input planes
# noinspection SpellCheckingInspection
# pieces_order = 'KQRBNPkqrbnp'   # 12x8x8
# castling_order = 'KQkq'       # 4x8x8
# fifty-move-rule             # 1x8x8
# en_passant               # 1x8x8

# Could someone write a quick documentation of the input planes?
# Here's what I think it is:
# The last 8 board positions. each one 8x8x12
# Current state, also 8x8x12
# Side to move, 8x8 constant
# Move number, 8x8 constant

# I think the move number is typo-d, it's using the halfmove clock instead of 50-move rule counter (which I assume is the intention).
# We also theoretically don't need the side to move because we can flip the board and invert the colors, so the side to move is always on the bottom and has king on the right. Alternatively we can augment the dataset x2 by applying this transformation, but I think the dimensionality reduction with x2 learning rate is at least equivalent (and probably better). (It doesn't work for Go because of the 7.5 komi rule)

# I think we're also missing castling.

# Another idea: shuffle training data to avoid overfitting to one game

# How is the policy vector represented?

# 8*8*12*8 = 6144

# Our state is represent by an 8x8x8 array
# Plane 0 represents pawns
# Plane 1 represents rooks
# Plane 2 represents knights
# Plane 3 represents bishops
# Plane 4 represents queens
# Plane 5 represents kings
# Plane 6 represents 1/fullmove number (needed for markov property)
# Plane 7 represents can-claim-draw
# White pieces have the value 1, black pieces are minus 1


# The Agent
# The agent is no longer a single piece, it's a chess player
# Its action space consist of 64x64=4096 actions:
# There are 8x8 = 64 piece from where a piece can be picked up
# And another 64 pieces from where a piece can be dropped.
# Of course, only certain actions are legal. Which actions are legal in a certain state is part of the environment (in RL, anything outside the control of the agent is considered part of the environment). We can use the python-chess package to select legal moves. (It seems that AlphaZero uses a similar approach https://ai.stackexchange.com/questions/7979/why-does-the-policy-network-in-alphazero-work)



12

# The output of the policy network is as described in the original paper:

# A move in chess may be described in two parts: selecting the piece to move, and then selecting among the legal moves for that piece. We represent the policy π(a|s) by a 8 × 8 × 73 stack of planes encoding a probability distribution over 4,672 possible moves. Each of the 8×8 positions identifies the square from which to “pick up” a piece. The first 56 planes encode possible ‘queen moves’ for any piece: a number of squares [1..7] in which the piece will be moved, along one of eight relative compass directions {N, NE, E, SE, S, SW, W, NW}. The next 8 planes encode possible knight moves for that piece. The final 9 planes encode possible underpromotions for pawn moves or captures in two possible diagonals, to knight, bishop or rook respectively. Other pawn moves or captures from the seventh rank are promoted to a queen.

# So each move selector scores the relative probability of selecting a piece in a given square and moving it in a specific way. For example, there is always one output dedicated to representing picking up the piece in A3 and moving it to A6. This representation includes selecting opponent pieces, selecting empty squares, making knight moves for rooks, making long diagonal moves for pawns. It also includes moves that take pieces off the board or through other blocking pieces.

# The typical branching factor in chess is around 35. The policy network described above always calculates discrete probabilities for 4672 moves.

# Clearly this can select many non-valid moves, if pieces are not available, or cannot move as suggested. In fact it does this all the time, even when fully trained, as nothing is ever learned about avoiding the non-valid moves during training - they do not receive positive or negative feedback, as there is never any experience gained relating to them. However, the benefit is that this structure is simple and fixed, both useful traits when building a neural network.

# The simple work-around is to filter out impossible moves logically, setting their effective probability to zero, and then re-normalise the probabilities for the remaining valid moves. That step involves asking the game engine for what the valid moves are - but that's fine, it's not "cheating".

# Whilst it might be possible to either have the agent learn to avoid non-valid moves, or some clever output structure that could only express valid moves, these would both distract from the core goal of learning how to play the game optimally.