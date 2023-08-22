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
np.random.seed(410)

training_index = "1440chunks_21_everything"

# PREP AND LOAD DATA

def prep_data(all_x,all_y,split):
  #all_x = torch.load(x)
  #all_y = torch.load(y)

  #all_x = torch.reshape(all_x, (all_x.shape[0],all_x.shape[2],all_x.shape[1]))

  data_x = all_x[:split,:,:]
  data_x_val = all_x[split:,:,:]
  data_y = all_y[:split,:]
  data_y_val = all_y[split:,:]

  planets_x = data_x[np.where((data_y[:,0]==1))]
  planets_y = data_y[np.where((data_y[:,0]==1))]
  planets_x_val = data_x_val[np.where((data_y_val[:,0]==1))]
  planets_y_val = data_y_val[np.where((data_y_val[:,0]==1))]
  
  eb_x = data_x[np.where((data_y[:,1]==1))]
  #eb_x = eb_x[::2,:,:]
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
  
data1x = torch.load("data_x_chunks_1440_flux2.pt")#[:,[0,1,2,3,4,5,7],:]
data2x = torch.load("data_x_chunks_1440_flux2_2.pt")#[:,[0,1,2,3,4,5,7],:]
data3x = torch.load("data_x_chunks_1440_flux2_3.pt")#[:,[0,1,2,3,4,5,7],:]
#data4x = torch.load("data_x_chunks_1440_4.pt")
#data5x = torch.load("data_x_chunks_1440_5.pt")
#data6x = torch.load("data_x_chunks_1440_6.pt")

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

#data_x, data_x_val, data_y, data_y_val = prep_data("data_x_chunks_720.pt","data_y_chunks_720.pt",180000)

#tshirt_pc_pre = np.where((data1y[:,0]==1))[0][:81]
#sorter = np.argsort(mask)
#tshirt_pc_post = sorter[np.searchsorted(mask, tshirt_pc_pre, sorter=sorter)]
#print(data1x[tshirt_pc_pre]==datax[tshirt_pc_post])

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

channels, n_out = 7,3

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
    #print(x.shape)
    #x = F.relu(self.linear1(x))
    #x = self.dropout(x)
    #x = F.relu(self.linear2(x))
    #x = self.dropout(x)
    #x = F.relu(self.linear3(x))
    #x = self.dropout(x)
    #print(x.shape)
    #x = F.relu(self.linear4(x))
    x = F.softmax(self.linear5(x),dim=1)
    return x
    
    

net = Classifier(channels, n_out)
#net.cuda()

net.load_state_dict(torch.load(f'training{training_index}/cp_{training_index}.ckpt',map_location=torch.device('cpu')))

# PERFORMANCE EVALUATION

planets = 0
eb = 0
other = 0
#nothing = 0 

y_true = np.array([])
y_pred = np.array([])
x_batch = np.array([])

net.eval()
for local_batch, local_labels in validation_generator:
  # Transfer to GPU
  #local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
  pred_y = net(local_batch[:,[0,1,2,3,4,5,7],:])
  pred_y = pred_y.cpu().detach().numpy()
  true_y = local_labels.numpy()
  if y_true.shape[0] == 0:
    y_true = true_y
    y_pred = pred_y
    x_batch = local_batch[:,[0,1,6],:]
  else:
    y_true = np.append(y_true, true_y, axis=0)
    y_pred = np.append(y_pred, pred_y, axis=0)
    x_batch = np.append(x_batch, local_batch[:,[0,1,6],:], axis=0)
  for i in pred_y:
      if np.argmax(i) == 0:
        planets = planets+1
      if np.argmax(i) == 1:
        eb = eb+1
      if np.argmax(i) == 2:
        other = other+1



print(planets,eb,other)#,nothing)


print(confusion_matrix(np.argmax(y_true,axis=1), np.argmax(y_pred,axis=1)))

print(precision_recall_fscore_support(np.argmax(y_true,axis=1), np.argmax(y_pred,axis=1), average=None))

a, b, c = roc_curve(y_true[:,0], y_pred[:,0])

thr = 3500

other = y_pred[y_pred[:,0]<c[thr]]
pc = y_pred[y_pred[:,0]>=c[thr]]

correct = 0
cont = 0
planet = 0
missed = 0 
others = 0

for i in range(len(y_pred)):
  if y_pred[i,0] < c[thr] and y_true[i,0] == 0:
    correct = correct + 1
    others = others + 1
  elif y_pred[i,0] >= c[thr] and y_true[i,0] == 1:
    correct = correct + 1
    planet = planet + 1
  elif y_pred[i,0] >= c[thr] and y_true[i,0] == 0:
    cont = cont + 1
  elif y_pred[i,0] < c[thr] and y_true[i,0] == 1:
    missed = missed + 1


accuracy = planet/len(y_true[y_true[:,0]==1])
contamination = cont/(len(pc))
print(f"percent of planets found: {accuracy}, contamination: {contamination}, planets found: {planet}, planets missed: {missed}, false positives: {cont}")

print(f"% of FP and EB removed: {(others)*100/(len(y_true[y_true[:,0]==0]))}")
print(f"num removed: {others}")

fp_as_pc = []
fp_as_eb = []
fp_as_fp = []


fp_as_pc_tics = []
fp_as_eb_tics = []

for i in range(len(y_true)):
    if np.argmax(y_true,axis=1)[i] == 2 and np.argmax(y_pred,axis=1)[i] == 0:
        fp_as_pc.append(i)
        fp_as_pc_tics.append(x_batch[i,6,0])
        
    if np.argmax(y_true,axis=1)[i] == 2 and np.argmax(y_pred,axis=1)[i] == 1:
        fp_as_eb.append(i)
        fp_as_eb_tics.append(x_batch[i,6,0])

    if np.argmax(y_true,axis=1)[i] == 2 and np.argmax(y_pred,axis=1)[i] == 2:
        fp_as_fp.append(i)


np.savetxt("fp_as_pc_tics.csv",np.array(fp_as_pc_tics),delimiter=",")
np.savetxt("fp_as_eb_tics.csv",np.array(fp_as_eb_tics),delimiter=",")

def make_plot(cols, rows, size, label,title):

    fig, axs = plt.subplots(cols,rows, figsize=size)
    fig.tight_layout(pad=-1.5)
    fig.suptitle(title, fontsize=20,y=0.99)
    plt.subplots_adjust(top=0.96,right=0.99)

    fig2, axs2 = plt.subplots(cols,rows, figsize=size)
    fig2.tight_layout(pad=-1.5)
    fig2.suptitle(title + " - background", fontsize=20,y=0.99) 
    plt.subplots_adjust(top=0.96,right=0.99)
    
    for row in range(rows):
        
        for col in range(cols):
            
            lc = np.random.choice(label,replace=False)
            
            axs[row][col].plot(x_batch[lc,0,:],".",markersize=1,color="indigo",label=f"{x_batch[lc,-1,0]}")
            axs[row][col].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            axs[row][col].legend()
            print(y_true[lc,2])
            axs2[row][col].plot(x_batch[lc,1,:],".",markersize=0.6,color="tab:orange")
            axs2[row][col].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
            
            label = np.delete(label,np.where(label==lc))
            #axs[row][col].axis("off")
            
    fig.savefig(title+".png") 
    fig2.savefig(title+"_bkg.png")    

make_plot(8,8,(15,18),fp_as_eb,"FP as EB")   
make_plot(8,8,(15,18),fp_as_pc,"FP as PC")
make_plot(8,8,(15,18),fp_as_fp,"FP as FP")

end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)
