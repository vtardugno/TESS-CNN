# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os 
import torch
import torch.nn as nn
from sklearn import model_selection


np.random.seed(410)

training_index = "9"

def prep_data(x,y,split):
  all_x = torch.load(x)
  all_y = torch.load(y)

  all_x = torch.reshape(all_x, (all_x.shape[0],all_x.shape[2],all_x.shape[1]))

  data_x = all_x[:split,:,:]
  data_x_val = all_x[split:,:,:]
  data_y = all_y[:split,:]
  data_y_val = all_y[split:,:]

  planets_x = data_x[np.where((data_y[:,0]==1))]
  planets_y = data_y[np.where((data_y[:,0]==1))]
  planets_x_val = data_x_val[np.where((data_y_val[:,0]==1))]
  planets_y_val = data_y_val[np.where((data_y_val[:,0]==1))]


  data_x = torch.cat((data_x,planets_x),0)
  data_y = torch.cat((data_y,planets_y),0)
  data_x_val = torch.cat((data_x_val,planets_x_val),0)
  data_y_val = torch.cat((data_y_val,planets_y_val),0)

  mask = np.array(range(len(data_x)))
  mask_val = np.array(range(len(data_x_val)))

  np.random.shuffle(mask)
  np.random.shuffle(mask_val)


  data_x = data_x[mask]
  data_y = data_y[mask]
  data_x_val = data_x_val[mask_val]
  data_y_val = data_y_val[mask_val]
  
  data_x = nn.functional.normalize(data_x,dim=-1)
  data_x_val = nn.functional.normalize(data_x_val,dim=-1)
  data_x = nn.functional.normalize(data_x,dim=-2)
  data_x_val = nn.functional.normalize(data_x_val,dim=-2)
  
  return data_x, data_x_val, data_y, data_y_val
  

#all_x = torch.load("data_x_s18_postFFT.pt")
#all_y = torch.load("data_y_s18_postFFT.pt")

#all_x = torch.reshape(all_x, (all_x.shape[0],all_x.shape[2],all_x.shape[1]))

#data_x = all_x[:820,:,:]
#data_x_val = all_x[820:,:,:]
#data_y = all_y[:820,:]
#data_y_val = all_y[820:,:]

#planets_x = data_x[np.where((data_y[:,0]==1))]
#planets_y = data_y[np.where((data_y[:,0]==1))]
#planets_x_val = data_x_val[np.where((data_y_val[:,0]==1))]
#planets_y_val = data_y_val[np.where((data_y_val[:,0]==1))]


#data_x = torch.cat((data_x,planets_x),0)
#data_y = torch.cat((data_y,planets_y),0)
#data_x_val = torch.cat((data_x_val,planets_x_val),0)
#data_y_val = torch.cat((data_y_val,planets_y_val),0)

#mask = np.array(range(len(data_x)))
#mask_val = np.array(range(len(data_x_val)))

#np.random.shuffle(mask)
#np.random.shuffle(mask_val)


#data_x = data_x[mask]
#data_y = data_y[mask]
#data_x_val = data_x_val[mask_val]
#data_y_val = data_y_val[mask_val]



# Normalize

#data_x = nn.functional.normalize(data_x,dim=-1)
#data_x_val = nn.functional.normalize(data_x_val,dim=-1)
#data_x = nn.functional.normalize(data_x,dim=-2)
#data_x_val = nn.functional.normalize(data_x_val,dim=-2)


data_x_18, data_x_val_18, data_y_18, data_y_val_18 = prep_data("data_x_s18_postFFT.pt","data_y_s18_postFFT.pt",820)
data_x_17, data_x_val_17, data_y_17, data_y_val_17 = prep_data("data_x_s17_postFFT.pt","data_y_s17_postFFT.pt",820)

print(data_x_val_18.shape,data_x_val_17.shape)

data_x_17 = data_x_17[:,:,(len(data_x_17[1,0,:])-1498)//2:-(len(data_x_17[1,0,:])-1498)//2]
data_x_val_17 = data_x_val_17[:,:,(len(data_x_val_17[1,0,:])-1498)//2:-(len(data_x_val_17[1,0,:])-1498)//2]

print(data_x_val_18.shape,data_x_val_17.shape)

data_x = torch.cat((data_x_18,data_x_17),0)
data_y = torch.cat((data_y_18,data_y_17),0)
data_x_val = torch.cat((data_x_val_18,data_x_val_17),0)
data_y_val = torch.cat((data_y_val_18,data_y_val_17),0)

mask = np.array(range(len(data_x)))
mask_val = np.array(range(len(data_x_val)))

np.random.shuffle(mask)
np.random.shuffle(mask_val)


data_x = data_x[mask]
data_y = data_y[mask]
data_x_val = data_x_val[mask_val]
data_y_val = data_y_val[mask_val]



# Build network


channels, n_out = 2,3

class Classifier(nn.Module):
  def __init__(self, channels, n_out):
    super(Classifier,self).__init__()
    self.conv1 = nn.Conv1d(channels, 128, kernel_size=5, padding="same")
    self.conv2 = nn.Conv1d(128, 64, kernel_size=5, padding="same")
    self.pool1 = nn.MaxPool1d(2)
    self.conv3 = nn.Conv1d(64, 32, kernel_size=5, padding="same")
    self.pool2 = nn.MaxPool1d(2)
    #self.conv3 = nn.Conv1d(64,100, kernel_size=5, padding="same")
    #self.pool2 = nn.MaxPool1d(2)
    #self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
    #self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
    #self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 2)
    self.linear1 = nn.Linear(32, 15)
    self.linear2 = nn.Linear(15, n_out)
    self.dropout = nn.Dropout(0.3) 

  def forward(self, x):
    #print(x.shape)
    x = F.relu(self.conv1(x))
    x = self.pool1(x)
    #print(x.shape)
    x = F.relu(self.conv2(x))
    x = self.pool2(x)
    x = F.relu(self.conv3(x))
    #x = self.pool2(x)
    #x = F.relu(self.conv3(x))
    x, _ = x.max(dim=-1) 
    #print(x.shape)
    x = F.relu(self.linear1(x))
    x = self.dropout(x)
    #print(x.shape)
    x = F.softmax(self.linear2(x),dim=1)
    return x
    
    

net = Classifier(channels, n_out)

# Give more weight to the planet candidates when calculating the loss

neg = torch.sum(data_y[:,2]!=1)
pos = torch.sum(data_y[:,2]==1)
total = neg+pos

w_0 = (1/neg)*(total/2)
w_1 = (1/pos)*(total/2)

weights = [w_1,w_0,w_0]
weights = torch.tensor(weights)

loss_function = nn.CrossEntropyLoss()#weight = weights)
#loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)#, weight_decay = 0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=500, gamma=0.1)


train_losses = []
val_losses = []


train_acc = []
val_acc = []


epochs = 300


# Train the network

for epoch in range(epochs):

    print(f"epoch: {epoch}")

    net.eval()
    with torch.no_grad():
      pred_y_val = net(data_x_val)
      val_loss = loss_function(pred_y_val, data_y_val)
      val_losses.append(val_loss.item())  
    
    net.train()
    pred_y = net(data_x)
    loss = loss_function(pred_y, data_y)
    train_losses.append(loss.item())
    
    
    train_corr = torch.argmax(pred_y,dim=1)==torch.argmax(data_y,dim=1)
    val_corr = torch.argmax(pred_y_val,dim=1)==torch.argmax(data_y_val,dim=1)
    
    #train_corr = len(torch.where(train_corr == True)[0])
    #val_corr = len(torch.where(val_corr == True)[0])
    train_corr = torch.sum(train_corr)
    val_corr = torch.sum(val_corr)
        
    train_acc.append(train_corr/len(pred_y))
    val_acc.append(val_corr/len(pred_y_val))

    net.zero_grad()
    loss.backward()

    optimizer.step()
    #scheduler.step()
    

# Create a directory and save the training results. If the directory already exists, add them there

try:
    os.mkdir(f"models/training{training_index}")
    checkpoint_path = f"models/training{training_index}/cp_{training_index}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
except:
    checkpoint_path = f"models/training{training_index}/cp_{training_index}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

torch.save(net.state_dict(), checkpoint_path)

    
# Plot loss and accuracy

import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_losses,label="train")
plt.plot(val_losses,label = "val")
plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig("loss.png")

plt.figure()
plt.plot(train_acc,label="train")
plt.plot(val_acc,label = "val")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig("acc.png")


pred_y = net(data_x)

planets = 0 
eb = 0
other = 0

for i in pred_y:
  if i[0] > 0.34:
    planets = planets+1
  if i[1] > 0.34:
    eb = eb+1
  if i[2] > 0.34:
    other = other+1
    
    
print(planets,eb,other)





