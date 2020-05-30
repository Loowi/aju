import os
import bz2
import glob

for name in glob.glob('C:\\Users\\Watson\\Projects\\remus\\input\\raw_games\\*'): 
    file = os.path.basename(name)
    archive_path = os.path.join('C:\\Users\\Watson\\Projects\\remus\\input\\raw_games',file)
    outfile_path = os.path.join('C:\\Users\\Watson\\Projects\\remus\\input\\games', file[:-4])
    with open(archive_path, 'rb') as source, open(outfile_path, 'wb') as dest:
        dest.write(bz2.decompress(source.read()))
