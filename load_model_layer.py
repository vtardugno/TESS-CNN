# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os 
import torch
import torch.nn as nn
import time
import pandas as pd
from sklearn import manifold, datasets
from sklearn.metrics import precision_recall_fscore_support, roc_curve, confusion_matrix


start_time = time.time()

torch.autograd.set_detect_anomaly(True)
np.random.seed(415)

#training_index = "1440chunks_21_everything_9"
training_index = "19"

# PREP AND LOAD DATA

# def prep_data(all_x,all_y,split,tics):
#   #all_x = torch.load(x)
#   #all_y = torch.load(y)

#   #all_x = torch.reshape(all_x, (all_x.shape[0],all_x.shape[2],all_x.shape[1]))

#   data_x = all_x[:split,:,:]
#   data_x_val = all_x[split:,:,:]
#   data_y = all_y[:split,:]
#   data_y_val = all_y[split:,:]

#   tics_train = tics[:split]
#   tics_val = tics[split:]

#   planet_mask = np.where((data_y[:,0]==1))
#   planet_mask_val = np.where((data_y_val[:,0]==1))
#   planets_x = data_x[planet_mask]
#   planets_y = data_y[planet_mask]
#   planets_x_val = data_x_val[planet_mask_val]
#   planets_y_val = data_y_val[planet_mask_val]
#   planets_tics = tics_train[planet_mask]
#   planets_tics_val = tics_val[planet_mask_val]
  
#   eb_mask = np.where((data_y[:,1]==1))
#   eb_mask_val = np.where((data_y_val[:,1]==1))
#   eb_x = data_x[eb_mask]
#   eb_y = data_y[eb_mask]
#   eb_x_val = data_x_val[eb_mask_val]
#   eb_y_val = data_y_val[eb_mask_val]
#   eb_tics = tics_train[eb_mask]
#   eb_tics_val = tics_val[eb_mask_val]

#   other_mask = np.where((data_y[:,2]==1))
#   other_mask_val = np.where((data_y_val[:,2]==1))
#   other_x = data_x[other_mask]
#   other_y = data_y[other_mask]
#   other_x_val = data_x_val[other_mask_val]
#   other_y_val = data_y_val[other_mask_val]
#   other_tics = tics_train[other_mask]
#   other_tics_val = tics_val[other_mask_val]

#   data_x = torch.cat((planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,eb_x,other_x,other_x,other_x,other_x),0)#nothing_x),0)
#   data_y = torch.cat((planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,eb_y,other_y,other_y,other_y,other_y),0)#,nothing_y),0)
#   data_x_val = torch.cat((planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,eb_x_val,other_x_val,other_x_val,other_x_val,other_x_val),0)#,nothing_x_val),0)
#   data_y_val = torch.cat((planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,eb_y_val,other_y_val,other_y_val,other_y_val,other_y_val),0)#,nothing_y_val),0)
  
#   tics_train = torch.cat((planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,eb_tics,other_tics,other_tics,other_tics,other_tics),0)#,nothing_y),0)
#   tics_val = torch.cat((planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,eb_tics_val,other_tics_val,other_tics_val,other_tics_val,other_tics_val),0)#,nothing_x_val),0)
  



#   mask = np.array(range(len(data_x)))
#   mask_val = np.array(range(len(data_x_val)))

#   #np.random.shuffle(mask)
#   #np.random.shuffle(mask_val)


#   #data_x = data_x[mask]
#   #data_y = data_y[mask]
#   #data_x_val = data_x_val[mask_val]
#   #data_y_val = data_y_val[mask_val]
  
 
#   print("planets", planets_y.shape, planets_y_val.shape, "eb", eb_y.shape, eb_y_val.shape, "fp", other_y.shape, other_y_val.shape, "tics", tics_train.shape, tics_val.shape)#, "nothing", nothing_y.shape, nothing_y_val.shape)
  
#   return data_x, data_x_val, data_y[:,:-1], data_y_val[:,:-1], tics_train, tics_val
  
# data1x = torch.load("data_x_final.pt")#[:,[0,1],:]
# data2x = torch.load("data_x_final_2.pt")#[:,[0,1],:]
# data3x = torch.load("data_x_final_3.pt")#[:,[0,1],:]


# data4x = torch.load("data_x_chunks_1440_no_tics.pt")
# data5x = torch.load("data_x_chunks_1440_no_tics_2.pt")
# data6x = torch.load("data_x_chunks_1440_no_tics_3.pt")

# data1y = torch.load("data_y_final.pt")
# data2y = torch.load("data_y_final_2.pt")
# data3y = torch.load("data_y_final_3.pt")

# id1 = torch.load("tic_ids_final.pt")
# id2 = torch.load("tic_ids_final_2.pt")
# id3 = torch.load("tic_ids_final_3.pt")

# sec1 = torch.load("sector_final.pt")
# sec2 = torch.load("sector_final_2.pt")
# sec3 = torch.load("sector_final_3.pt")

# datax = torch.cat((data1x,data2x,data3x),0)
# datay = torch.cat((data1y,data2y,data3y),0)
# ids = torch.cat((id1,id2,id3),0)
# secs = torch.cat((sec1,sec2,sec3),0)

#datax2 = torch.cat((data4x,data5x,data6x),0)


# mask = np.array(range(len(datax)))
# np.random.shuffle(mask)
# datax = datax[mask]
# datay = datay[mask]
# ids = ids[mask]

#datax2 = datax2[mask]

#data_x, data_x_val, data_y, data_y_val, tics_train, tics_val = prep_data(datax, datay ,225000,ids)

# data_x_val = datax[205000:]
# data_y_val = datay[205000:] 
# tics_val = ids[205000:]
# secs_val = secs[205000:]

# print(np.unique(secs_val))

# validation_set = torch.utils.data.TensorDataset(data_x_val,data_y_val,tics_val,secs_val)
# validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = 1024, shuffle=True)


#data_x_2 , _, _, _, _, _ = prep_data(datax2, datay ,225000,ids)

# mask = np.array(range(len(data_x)))
# mask_val = np.array(range(len(data_x_val)))

# np.random.shuffle(mask)
# np.random.shuffle(mask_val)


# data_x = data_x[mask]
# data_y = data_y[mask]
# data_x_val = data_x_val[mask_val]
# data_y_val = data_y_val[mask_val]

# tics_train = tics_train[mask]
# tics_val = tics_val[mask_val]

#data_x_2 = data_x_2[mask]


#val_tensor = torch.load("val_tensor_all.pt")
tensor_1 = torch.load("test_tensor_1.pt")
tensor_2 = torch.load("test_tensor_2.pt")
tensor_3 = torch.load("test_tensor_3.pt")



datax = torch.cat((tensor_1[0],tensor_2[0],tensor_3[0]),0)#[207000:,:,:]
datay = torch.cat((tensor_1[1],tensor_2[1],tensor_3[1]),0)#[207000:,:]
ids = torch.cat((tensor_1[2], tensor_2[2],tensor_3[2]),0)#[207000:]
secs = torch.cat((tensor_1[3], tensor_2[3],tensor_3[3]),0)#[207000:]



#datax = val_tensor[0]#,tensor_2[0],tensor_3[0]),0)[207000:,:,:]
#datay = val_tensor[1]#,tensor_2[1],tensor_3[1]),0)[207000:,:]
#ids = val_tensor[2]#, tensor_2[2],tensor_3[2]),0)[207000:]
#secs = val_tensor[3]#, tensor_2[3],tensor_3[3]),0)[207000:]


#validation_set = torch.utils.data.TensorDataset(test_tensor[0],test_tensor[1],test_tensor[2],test_tensor[3])
validation_set = torch.utils.data.TensorDataset(datax,datay,ids,secs)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = 1024, shuffle=True)


# LOAD MODEL

channels, n_out = 6,3

class Classifier(nn.Module):
    def __init__(self, channels, n_out):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv1d(channels, 32, kernel_size=5, padding=2)
        self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding=2)
        self.layernorm1 = nn.LayerNorm((32, 1440))  # Adjusted LayerNorm dimensions
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding=2)
        self.conv2_2 = nn.Conv1d(64, 64, kernel_size=5, padding=2)
        self.layernorm2 = nn.LayerNorm((64, 720))  # Adjusted LayerNorm dimensions
        self.pool2 = nn.MaxPool1d(2)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.conv3_2 = nn.Conv1d(128, 128, kernel_size=5, padding=2)
        self.layernorm3 = nn.LayerNorm((128, 360))  # Adjusted LayerNorm dimensions
        self.pool3 = nn.MaxPool1d(2)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=5, padding=2)
        self.conv4_2 = nn.Conv1d(256, 256, kernel_size=5, padding=2)
        self.layernorm4 = nn.LayerNorm((256, 180))  # Adjusted LayerNorm dimensions
        self.pool4 = nn.MaxPool1d(2)
        self.conv5 = nn.Conv1d(256, 128, kernel_size=5, padding=2)  # new
        self.conv6 = nn.Conv1d(128, 64, kernel_size=5, padding=2)  # new
        self.layernorm5 = nn.LayerNorm((64, 90))  # Adjusted LayerNorm dimensions
        self.pool5 = nn.MaxPool1d(2)  # new
        self.conv7 = nn.Conv1d(64, 32, kernel_size=5, padding=2)  # new
        self.conv8 = nn.Conv1d(32, 16, kernel_size=5, padding=2)  # new
        self.layernorm6 = nn.LayerNorm((16, 45))  # Adjusted LayerNorm dimensions
        self.pool6 = nn.MaxPool1d(2)  # new

        self.linear5 = nn.Linear(16, n_out)
        self.dropout = nn.Dropout(0.1) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.layernorm1(self.conv1_2(x)))
        x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.layernorm2(self.conv2_2(x)))
        x = self.dropout(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.layernorm3(self.conv3_2(x)))
        x = self.dropout(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.layernorm4(self.conv4_2(x)))
        x = self.dropout(x)
        x = self.pool4(x)
        x = F.relu(self.conv5(x))  # new starts here
        x = F.relu(self.layernorm5(self.conv6(x)))
        x = self.dropout(x)
        x = self.pool5(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.layernorm6(self.conv8(x)))
        x = self.pool6(x)  # new ends here

        x, _ = x.max(dim=-1)
        x = F.softmax(self.linear5(x), dim=1)
        return x
    

net = Classifier(channels, n_out)
#net.cuda()
# net2 = Classifier(1,n_out)

net.load_state_dict(torch.load(f'training{training_index}/best_{training_index}.ckpt',map_location=torch.device('cpu')))
#net2.load_state_dict(torch.load('training1440chunks_21_flux_only_3/cp_1440chunks_21_flux_only_3.ckpt',map_location=torch.device('cpu')))

# PERFORMANCE EVALUATION

y_true = np.array([])
y_pred = np.array([])
x_batch = np.array([])
tic_ids = np.array([])
sectors = np.array([])

# y_pred2 = np.array([])

net.eval()
for local_batch, local_labels, tics, sect in validation_generator:
  # Transfer to GPU
  #local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
  pred_y = net(local_batch)
  pred_y = pred_y.cpu().detach().numpy()

  # pred_y2 = net2(local_batch[:,[0],:])
  # pred_y2 = pred_y2.cpu().detach().numpy()

  true_y = local_labels.numpy()
  if y_true.shape[0] == 0:
    y_true = true_y
    y_pred = pred_y
    # y_pred2 = pred_y2
    x_batch = local_batch[:,[0],:]
    tic_ids = tics.numpy()
    sectors = sect.numpy()
  else:
    y_true = np.append(y_true, true_y, axis=0)
    y_pred = np.append(y_pred, pred_y, axis=0)
    # y_pred2 = np.append(y_pred2, pred_y2, axis=0)
    x_batch = np.append(x_batch, local_batch[:,[0],:], axis=0)
    tic_ids = np.append(tic_ids, tics, axis=0)
    sectors = np.append(sectors, sect, axis=0)

print(np.unique(sectors))

# np.save(f"y_true_test_s50_s53_s54_{training_index}.npy",y_true)
# np.save(f"y_pred_test_s50_s53_s54_{training_index}.npy",y_pred)
# np.save(f"y_tic_test_s50_s53_s54_{training_index}.npy",tic_ids)
# np.save(f"y_sec_test_s50_s53_s54_{training_index}.npy",sectors)
#np.save(f"x_test_s50_s53_s54_{training_index}.npy",x_batch)
np.save(f"y_true_test_final_{training_index}.npy",y_true)
np.save(f"y_pred_test_final_{training_index}.npy",y_pred)
np.save(f"tic_test_final_{training_index}.npy",tic_ids)
np.save(f"secs_test_final_{training_index}.npy",sectors)
