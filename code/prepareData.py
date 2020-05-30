import glob
from multiprocessing import Pool, cpu_count
import pandas as pd
from os import getpid


def process(file):
    print("I'm process", getpid())
    data = pd.read_csv(file, header = None)
    print('Success!')
    return data.sum(axis=1)


if __name__ == '__main__':
    print(cpu_count())
    files = glob.glob('test*.txt')
    pool = Pool(6)
    results = pool.map(process, files)
    pool.close()
    print(results)


# with Pool(3) as p:
#     print(123)
#     # print(p.map(f, [1, 2, 3]))
#     result = p.map(process, files)



# # result.wait()1
# p.close()
# # p.join() # Wait for all child processes to close.






# for f in glob.glob('test*.txt'):
#     print(f)    
#     # result = p.apply_async(process, [f]) 





# from multiprocessing import Pool, cpu_count

# kala = []

# def test(team_name, team_data):
#     # do some fancy model fitting here
#     return model

# with Pool(cpu_count() - 1) as pool:
#     models = pool.starmap(test, teams)
#     # res = pd.concat(p.starmap(update_table, zip(rows)))


