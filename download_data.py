from astroquery.mast import Catalogs
import numpy as np
from torch.utils.data import Dataset, DataLoader

data = np.genfromtxt('ground_truth_2.csv', delimiter=',', usecols=(1,), dtype=np.int64)
label = np.genfromtxt('ground_truth_2.csv', delimiter=',', usecols=(3,), dtype=str)

label_index = { 'planet':0, 'EB':1, 'other':2}

labels = np.array([label_index[label[i]] for i in range(len(label))])

catalog_data = Catalogs.query_criteria(catalog="Tic",ID=data)

field_name = ['Vmag', 'Teff', 'rad']

fields = np.stack([catalog_data[field_name[i]].data.data for i in range(len(field_name))])

mask_nan = np.any(np.isnan(fields), axis=0)

clean_data = fields[:,~mask_nan]


np.savez('./test.npz', tic_ids = data[~mask_nan], fields = clean_data, labels = labels[~mask_nan])

class TessMetaDataset(Dataset):
    def __init__(self, path: str = "./"):
        data = np.load(path)
        self.tic_ids = data['tic_ids']
        self.fields = data['fields']
        self.labels = data['labels']

    def __len__(self):
        return len(self.tic_ids)

    def __getitem__(self, idx):
        return self.tic_ids[idx], self.fields[:,idx], self.labels[idx]

        

