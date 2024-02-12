from astroquery.mast import Catalogs
import numpy as np
from torch.utils.data import Dataset, DataLoader

sectors = [2,3,5,6,17,19,45,4,18,10,11,34,35,36,37,38,42]

for sec in sectors:
    data = np.genfromtxt(f'ground_truth/ground_truth/ground_truth_{sec}.csv', delimiter=',', usecols=(1,), skip_header = 1, dtype=np.int64)
    label = np.genfromtxt(f'ground_truth/ground_truth/ground_truth_{sec}.csv', delimiter=',', usecols=(3,), skip_header = 1,dtype=str)

    label_index = { 'planet':0, 'EB':1, 'other':2}

    labels = np.array([label_index[label[i]] for i in range(len(label))])

    catalog_data = Catalogs.query_criteria(catalog="Tic",ID=data)

    field_name = ['Vmag', 'Teff', 'rad']

    fields = np.stack([catalog_data[field_name[i]].data.data for i in range(len(field_name))])

    mask_nan = np.any(np.isnan(fields), axis=0)

    clean_data = fields[:,~mask_nan]

    np.savez(f'./{sec}_meta.npz', tic_ids = data[~mask_nan], fields = clean_data, labels = labels[~mask_nan])


test = np.load('./{sec}_meta.npz')
print(test)

# class TessMetaDataset(Dataset):
#     def __init__(self, path: str = "./"):
#         data = np.load(path)
#         self.tic_ids = data['tic_ids']
#         self.fields = data['fields']
#         self.labels = data['labels']

#     def __len__(self):
#         return len(self.tic_ids)

#     def __getitem__(self, idx):
#         return self.tic_ids[idx], self.fields[:,idx], self.labels[idx]


# dataset = TessMetaDataset('./test.npz')
# train_loader = DataLoader(dataset, batch_size=32, shuffle=True)


