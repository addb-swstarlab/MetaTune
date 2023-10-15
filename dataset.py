from torch.utils.data import Dataset

class AttnDataset(Dataset):
    def __init__(self, X, y):
        super(AttnDataset, self).__init__()
        self.X = X.unsqueeze(axis=-1)
        self.y = y.unsqueeze(axis=-1)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])

class SingleDataset(Dataset):
    def __init__(self, X, y):
        super(SingleDataset, self).__init__()
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.y[idx])