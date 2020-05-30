import glob, pickle, chess, logging
from multiprocessing import Pool, cpu_count
from os import getpid
import chess.pgn
import pandas as pd  

logger = logging.getLogger('Main')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s', datefmt='%m/%d/%Y %H:%M:%S')
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel(logging.INFO)

def process(file):
    logger = logging.getLogger('Main.ProcessFiles')
    logger.info('Started process: %s', getpid())

    pgn = open(file)
    games=[]
    i = 0
    while True:
        game=chess.pgn.read_game(pgn)
        if game == None:
            break
        games.append(game)
        i = i+1
        if i%10000 == 0:
            logger.info('Process %s processed %s games', getpid(), i)     

    logger.info('Process %s collected %s games', getpid(), len(games))
    
    
    positionData = []

    for gameNum, game in enumerate(games):
        board = game.board()
        winlose = game.headers["Result"]
        outcomes = {'0-1': -1,'1-0': 1, '1/2-1/2': 0} 
        win = outcomes[winlose]

        id = game.headers["FICSGamesDBGameNo"]
        if gameNum%10000 == 0:
            logger.info('Process %s has moves for %s games', getpid(), gameNum) 
                
        for moveNum, move in enumerate(game.mainline_moves()):
            board.push(move) 
            fen = board.fen()
            
            # Store the result
            positionData.append([id, gameNum, moveNum, fen, win])
         
    df = pd.DataFrame(positionData, columns=['ID','Game','Move','Fen','Result'])

    df.replace({'A': {0: 100, 4: 400}})

    logger.info('Process %s has finished', getpid()) 
    
    return df

def main():
    logger.info('Started the process')
    logger.info('Number of CPUs: %s', cpu_count()) 
    files = glob.glob('C:\\Users\\Watson\\Projects\\remus\\input\\games\\*')
    logger.info('Number of files to be processed: %s', len(files))
        
    pool = Pool(6)
    results = pool.map(process, files)
    pool.close()

    t = pd.concat(results)

    logger.info('Started pickling the results')
    with open('allGames.pickle', 'wb') as f:
        pickle.dump(t, f)

if __name__ == '__main__':
    main()



