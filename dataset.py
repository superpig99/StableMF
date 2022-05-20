import numpy as np
import torch
from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, filepath: str, timestamp: bool=False):
        self.ruid = []
        self.riid = []
        self.rating = []
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if timestamp:
                for line in lines:
                    ruid, riid, rating, _ = line.strip().split('\t')
                    ruid = int(ruid)
                    riid = int(riid)
                    # ruid = torch.tensor(int(ruid))
                    # riid = torch.tensor(int(riid))
                    rating = torch.tensor(float(rating))
                    self.ruid.append(ruid)
                    self.riid.append(riid)
                    self.rating.append(rating)
            else:
                for line in lines:
                    ruid, riid, rating = line.strip().split('\t')
                    ruid = int(ruid)
                    riid = int(riid)
                    # ruid = torch.tensor(int(ruid))
                    # riid = torch.tensor(int(riid))
                    rating = torch.tensor(float(rating))
                    self.ruid.append(ruid)
                    self.riid.append(riid)
                    self.rating.append(rating)
            f.close()

    def __getitem__(self, index):
        x = np.zeros(2, dtype=np.int64)
        # x = torch.zeros(2, dtype=torch.int64)
        x[0] = self.ruid[index]
        x[1] = self.riid[index]
        y = self.rating[index]
        return x, y
        #return (self.ruid[index], self.riid[index]), self.rating[index]

        # (pu, qi, rating) = self.pu_qi_rating[index]
        # x = torch.from_numpy(np.vstack((pu, qi)))
        # y = torch.tensor(rating)
        # return x, y
    
    def __len__(self):
        return len(self.rating)
