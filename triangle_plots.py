import numpy as np
import matplotlib.pyplot as plt

from ternary_diagram import TernaryDiagram
import pandas as pd
import torch.nn.functional as F
import os 
import torch
import torch.nn as nn
import time
import pandas as pd
from sklearn import manifold, datasets
from sklearn.metrics import precision_recall_fscore_support, roc_curve, confusion_matrix

training_indices = ["1440chunks_21_everything","1440chunks_21_everything_except_bkg","1440chunks_21_flux1_only","1440chunks_21_everything_except_moms","1440chunks_21_everything_except_flux2"]
#training_indices = ["1440chunks_21_everything","1440chunks_21_flux1_only"]

def prep_data(all_x,all_y,split):

    data_x = all_x[:split,:,:]
    data_x_val = all_x[split:,:,:]
    data_y = all_y[:split,:]
    data_y_val = all_y[split:,:]

    planets_x = data_x[np.where((data_y[:,0]==1))]
    planets_y = data_y[np.where((data_y[:,0]==1))]
    planets_x_val = data_x_val[np.where((data_y_val[:,0]==1))]
    planets_y_val = data_y_val[np.where((data_y_val[:,0]==1))]
    
    eb_x = data_x[np.where((data_y[:,1]==1))]
    eb_y = data_y[np.where((data_y[:,1]==1))]
    #eb_y = eb_y[::2,:]
    eb_x_val = data_x_val[np.where((data_y_val[:,1]==1))]
    #eb_x_val = eb_x_val[::2,:,:]
    eb_y_val = data_y_val[np.where((data_y_val[:,1]==1))]
    #eb_y_val = eb_y_val[::2,:]

    other_x = data_x[np.where((data_y[:,2]==1))]
    # other_x = other_x[::2,:,:]
    other_y = data_y[np.where((data_y[:,2]==1))]
    #other_y = other_y[::2,:]

    other_x_val = data_x_val[np.where((data_y_val[:,2]==1))]
    #other_x_val = other_x_val[::2,:,:]
    other_y_val = data_y_val[np.where((data_y_val[:,2]==1))]
    #other_y_val = other_y_val[::2,:]

    data_x = torch.cat((planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,eb_x,other_x,other_x,other_x,other_x),0)#nothing_x),0)
    data_y = torch.cat((planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,eb_y,other_y,other_y,other_y,other_y),0)#,nothing_y),0)
    data_x_val = torch.cat((planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,eb_x_val,other_x_val,other_x_val,other_x_val,other_x_val),0)#,nothing_x_val),0)
    data_y_val = torch.cat((planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,eb_y_val,other_y_val,other_y_val,other_y_val,other_y_val),0)#,nothing_y_val),0)


    mask = np.array(range(len(data_x)))
    mask_val = np.array(range(len(data_x_val)))

    
    
    print("planets", planets_y.shape, planets_y_val.shape, "eb", eb_y.shape, eb_y_val.shape, "fp", other_y.shape, other_y_val.shape)#, "nothing", nothing_y.shape, nothing_y_val.shape)
    
    return data_x, data_x_val, data_y[:,:-1], data_y_val[:,:-1]
    

def run_model(training_index):

    torch.autograd.set_detect_anomaly(True)
    np.random.seed(410)

    
    data1x = torch.load("data_x_chunks_1440_flux2.pt")#[:,[0,1,2,3,4,5,7],:]
    data2x = torch.load("data_x_chunks_1440_flux2_2.pt")#[:,[0,1,2,3,4,5,7],:]
    data3x = torch.load("data_x_chunks_1440_flux2_3.pt")#[:,[0,1,2,3,4,5,7],:]

    data1y = torch.load("data_y_chunks_1440_flux2.pt")
    data2y = torch.load("data_y_chunks_1440_flux2_2.pt")
    data3y = torch.load("data_y_chunks_1440_flux2_3.pt")
    #data4y = torch.load("data_y_chunks_1440_4.pt")
    #data5y = torch.load("data_y_chunks_1440_5.pt")
    #data6y = torch.load("data_y_chunks_1440_6.pt")

    datax = torch.cat((data1x,data2x,data3x),0)
    datay = torch.cat((data1y,data2y,data3y),0)

    mask = np.array(range(len(datax)))
    np.random.shuffle(mask)
    datax = datax[mask]
    datay = datay[mask]

    data_x, data_x_val, data_y, data_y_val = prep_data(datax, datay ,225000)


    mask = np.array(range(len(data_x)))
    mask_val = np.array(range(len(data_x_val)))

    np.random.shuffle(mask)
    np.random.shuffle(mask_val)


    data_x = data_x[mask]
    data_y = data_y[mask]
    data_x_val = data_x_val[mask_val]
    data_y_val = data_y_val[mask_val]

    print(data_x.shape,data_x_val.shape)

    training_set = torch.utils.data.TensorDataset(data_x,data_y)
    training_generator = torch.utils.data.DataLoader(training_set, batch_size = 1024, shuffle=True)

    validation_set = torch.utils.data.TensorDataset(data_x_val,data_y_val)
    validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = 1024, shuffle=True)

    #torch.reshape(data_x,(data_x.shape[0],1,data_x.shape[1])) #only use this when you have 1 channel only
    #torch.reshape(data_x_val,(data_x_val.shape[0],1,data_x_val.shape[1]))
    print(data_x.shape,data_x_val.shape)

    # LOAD MODEL

    if training_index == "1440chunks_21_everything":
        channels, n_out = 7,3
    elif training_index == "1440chunks_21_flux1_only":
        channels, n_out = 1,3
    elif training_index == "1440chunks_21_everything_except_moms":
        channels, n_out = 5,3

    else:
        channels, n_out = 6,3

    class Classifier(nn.Module):
        def __init__(self, channels, n_out):
            super(Classifier,self).__init__()
            self.conv1 = nn.Conv1d(channels, 32, kernel_size=5, padding="same")
            self.conv1_2 = nn.Conv1d(32, 32, kernel_size=5, padding="same")
            self.pool1 = nn.MaxPool1d(2)
            self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding="same")
            self.conv2_2 = nn.Conv1d(64, 64, kernel_size=5, padding="same")
            self.pool2 = nn.MaxPool1d(2)
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding="same")
            self.conv3_2 = nn.Conv1d(128, 128, kernel_size=5, padding="same")
            self.pool3 = nn.MaxPool1d(2)
            self.conv4 = nn.Conv1d(128,256, kernel_size=5, padding="same")
            self.conv4_2 = nn.Conv1d(256,256, kernel_size=5, padding="same")
            self.pool4 = nn.MaxPool1d(2)
            self.conv5 = nn.Conv1d(256,128, kernel_size=5, padding="same") #new
            self.conv6 = nn.Conv1d(128,64, kernel_size=5, padding="same") #new
            self.pool5 = nn.MaxPool1d(2)#new
            self.conv7 = nn.Conv1d(64,32, kernel_size=5, padding="same") #new
            self.conv8 = nn.Conv1d(32,16, kernel_size=5, padding="same") #new
            self.pool6 = nn.MaxPool1d(2)#new


            #self.linear1 = nn.Linear(256, 128)
            #self.linear2 = nn.Linear(128, 64)
            #self.linear3 = nn.Linear(64, 32)
            #self.linear4 = nn.Linear(32, 16)
            self.linear5 = nn.Linear(16, n_out)
            self.dropout = nn.Dropout(0.3) 

        def forward(self, x):
            #print(x.shape)
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv1_2(x))
            x = self.dropout(x)
            x = self.pool1(x)
            #print(x.shape)
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv2_2(x))
            x = self.dropout(x)
            x = self.pool2(x)
            x = F.relu(self.conv3(x))
            x = F.relu(self.conv3_2(x))
            x = self.dropout(x)
            x = self.pool3(x)
            x = F.relu(self.conv4(x))
            x = F.relu(self.conv4_2(x))
            x = self.dropout(x)
            x = self.pool4(x)
            x = F.relu(self.conv5(x)) #new starts here
            x = F.relu(self.conv6(x))
            x = self.dropout(x)
            x = self.pool5(x)
            x = F.relu(self.conv7(x)) 
            x = F.relu(self.conv8(x))
            x = self.pool6(x) #new ends here

            x, _ = x.max(dim=-1) 
            x = F.softmax(self.linear5(x),dim=1)
            return x
            

    net = Classifier(channels, n_out)
    #net.cuda()

    net.load_state_dict(torch.load(f'training{training_index}/cp_{training_index}.ckpt',map_location=torch.device('cpu')))
    y_true = np.array([])
    y_pred = np.array([])
    #x_batch = np.array([])

    net.eval()
    for local_batch, local_labels in validation_generator:
    # Transfer to GPU
    #local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
        if training_index == "1440chunks_21_everything":
            pred_y = net(local_batch[:,[0,1,2,3,4,5,7],:])
        if training_index == "1440chunks_21_flux1_only":
            pred_y = net(local_batch[:,[0],:])
        if training_index == "1440chunks_21_everything_except_bkg":
            pred_y = net(local_batch[:,[0,2,3,4,5,7],:])
        if training_index == "1440chunks_21_everything_except_flux2":
            pred_y = net(local_batch[:,[0,1,2,3,4,5],:])
        if training_index == "1440chunks_21_everything_except_moms":
            pred_y = net(local_batch[:,[0,1,4,5,7],:])    

        pred_y = pred_y.cpu().detach().numpy()
        true_y = local_labels.numpy()
        if y_true.shape[0] == 0:
            y_true = true_y
            y_pred = pred_y
            #x_batch = local_batch[:,[0,1,6],:]
        else:
            y_true = np.append(y_true, true_y, axis=0)
            y_pred = np.append(y_pred, pred_y, axis=0)
            #x_batch = np.append(x_batch, local_batch[:,[0,1,6],:], axis=0)

    return y_true,y_pred


def make_plot(training_index):
    fig, axes = plt.subplots(1, 3, dpi=72, facecolor="white", figsize=(10, 4))

    axes[0].set_title("planets")

    y_true, y_pred = (run_model(training_index))

    td = TernaryDiagram(["planet", "eb", "fp"], ax = axes[0])

    td.scatter(vector=y_pred[np.where(y_true[:,0] == 1)], marker=".", color = "red", s=15,alpha=0.1,zorder=30)

    axes[1].set_title("eb")
    td = TernaryDiagram(["planet", "eb", "fp"],ax = axes[1])
    td.scatter(vector=y_pred[np.where(y_true[:,1] == 1)], marker=".", color = "blue", s=15,alpha=0.1,zorder=30)


    axes[2].set_title("fp")
    td = TernaryDiagram(["planet", "eb", "fp"],ax = axes[2])

    td.scatter(vector=y_pred[np.where(y_true[:,2] == 1)], marker=".", color = "green", s=15,alpha=0.1,zorder=30)

    plt.savefig(f"{training_index}.png")


for i in training_indices:
    make_plot(i)

