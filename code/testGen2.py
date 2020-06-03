class testgen(Sequence):

    def __init__(self, files, num_files, batch_size):
        self.files = files
        self.num_files = num_files
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)

    def __getitem__(self, idx):
        listPandaFiles = [self.files.pop(0) for i in range(self.num_files)]
        df_new = pd.concat(listPandaFiles)
        print(df_new)
        return df_new