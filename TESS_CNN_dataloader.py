# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os 
import torch
import torch.nn as nn

torch.autograd.set_detect_anomaly(True)
np.random.seed(410)

training_index = "15"

def prep_data(x,y,split):
  all_x = torch.load(x)
  all_y = torch.load(y)

  #all_x = torch.reshape(all_x, (all_x.shape[0],all_x.shape[2],all_x.shape[1]))

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
  
  data_x[:,0,:] = nn.functional.normalize(data_x[:,0,:],dim=1)
  data_x_val[:,0,:] = nn.functional.normalize(data_x_val[:,0,:],dim=1)
  data_x[:,1,:] = nn.functional.normalize(data_x[:,1,:],dim=1)
  data_x_val[:,1,:] = nn.functional.normalize(data_x_val[:,1,:],dim=1)
  
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


#data_x_18, data_x_val_18, data_y_18, data_y_val_18 = prep_data("data_x_s18_clean.pt","data_y_s18_clean.pt",820)
#ddata_x_17, data_x_val_17, data_y_17, data_y_val_17 = prep_data("data_x_s17_clean.pt","data_y_s17_clean.pt",820)


data_x, data_x_val, data_y, data_y_val = prep_data("data_x_6-17-18-19-45.pt","data_y_6-17-18-19-45.pt",4000)
#data_x, data_x_val, data_y, data_y_val = prep_data("fourier_data_x_6-17-18-19-45.pt","fourier_data_y_6-17-18-19-45.pt",4000)


#data_x = torch.cat((data_x,data_x_val),0)
#data_y = torch.cat((data_y,data_y_val),0)

mask = np.array(range(len(data_x)))
mask_val = np.array(range(len(data_x_val)))

np.random.shuffle(mask)
np.random.shuffle(mask_val)


data_x = data_x[mask]
data_y = data_y[mask]
data_x_val = data_x_val[mask_val]
data_y_val = data_y_val[mask_val]

#data_x = torch.view_as_real(data_x)
#data_x_val = torch.view_as_real(data_x_val)

print(data_x.shape,data_x_val.shape)

training_set = torch.utils.data.TensorDataset(data_x,data_y)
training_generator = torch.utils.data.DataLoader(training_set,  batch_size = 256, shuffle=True)

validation_set = torch.utils.data.TensorDataset(data_x_val,data_y_val)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size = 256, shuffle=True)

#data_x = torch.flatten(data_x,start_dim = 2, end_dim = 3)
#data_x_val = torch.flatten(data_x_val,start_dim = 2, end_dim = 3)


#data_x = data_x[:3000,:,:].cuda()

#data_y = data_y[:3000,:].cuda()

#data_x_val = data_x_val[:1000,:,:].cuda()

#data_y_val = data_y_val[:1000,:].cuda()


print(data_x.shape,data_x_val.shape)


#for i in np.array([14,67]):
  #plt.figure()
  #plt.plot(data_x.cpu()[i,0,:])
  #plt.savefig(f"LC{i}")
  #plt.figure()
  #plt.plot(data_x.cpu()[i,1,:])
  #plt.savefig(f"BKG{i}")


# Build network


channels, n_out = 2,3

class Classifier(nn.Module):
  def __init__(self, channels, n_out):
    super(Classifier,self).__init__()
    self.conv1 = nn.Conv1d(channels, 32, kernel_size=5, padding="same")
    self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding="same")
    self.pool1 = nn.MaxPool1d(2)
    self.conv3 = nn.Conv1d(64, 128, kernel_size=5, padding="same")
    self.conv4 = nn.Conv1d(128,256, kernel_size=5, padding="same")
    self.pool2 = nn.MaxPool1d(2)

    #self.pool2 = nn.MaxPool1d(2)
    #self.conv3 = nn.Conv1d(in_channels=32, out_channels=16, kernel_size=3)
    #self.conv4 = nn.Conv1d(in_channels=16, out_channels=16, kernel_size=3)
    #self.pool2 = nn.MaxPool1d(kernel_size = 2, stride = 2)
    self.linear1 = nn.Linear(256, 90)
    #self.linear2 = nn.Linear(128, 32)
    self.linear3 = nn.Linear(90, n_out)
    self.dropout = nn.Dropout(0.3) 

  def forward(self, x):
    #print(x.shape)
    x = F.relu(self.conv1(x))
    
    #print(x.shape)
    x = F.relu(self.conv2(x))
    x = self.pool1(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool2(x)
    #x = self.pool2(x)
    #x = F.relu(self.conv3(x))
    x, _ = x.max(dim=-1) 
    #print(x.shape)
    x = F.relu(self.linear1(x))
    x = self.dropout(x)
    #x = F.relu(self.linear2(x))
    #x = self.dropout(x)
    #print(x.shape)
    x = F.softmax(self.linear3(x),dim=1)
    return x
    
    

net = Classifier(channels, n_out)
net.cuda()

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
optimizer = torch.optim.Adam(net.parameters(),lr=0.0005)#, weight_decay = 0.9)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.1)


train_losses = []
val_losses = []


train_acc = []
val_acc = []


epochs = 500


# Train the network

for epoch in range(epochs):

    print(f"epoch: {epoch}")
    
    net.eval()
    
    loss_loop = []
    val_loss_loop = []
    acc_loop = []
    val_acc_loop  = []

    for local_batch, local_labels in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
        net.train()
        pred_y = net(local_batch)
        loss = loss_function(pred_y, local_labels)
        loss_loop.append(loss.item())
        train_corr_pre = torch.argmax(pred_y,dim=1)==torch.argmax(local_labels,dim=1)
        train_corr = torch.sum(train_corr_pre)
        acc_loop.append((train_corr/len(pred_y)).cpu())
        optimizer.step()

    for local_batch, local_labels in validation_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
        with torch.no_grad():
             pred_y_val = net(local_batch)
             val_loss = loss_function(pred_y_val, local_labels)
             val_loss_loop.append(val_loss.item())
             val_corr_pre = torch.argmax(pred_y_val,dim=1)==torch.argmax(local_labels,dim=1)
             val_corr = torch.sum(val_corr_pre)
             val_acc_loop.append((val_corr/len(pred_y_val)).cpu())


    val_losses.append(np.average(np.array(val_loss_loop)))
    train_losses.append(np.average(np.array(loss_loop)))
        
    train_acc.append(np.average(np.array(acc_loop)))
    val_acc.append(np.average(np.array(val_acc_loop)))

    net.zero_grad()
    loss.backward()


    #scheduler.step()
    

# Create a directory and save the training results. If the directory already exists, add them there

try:
    os.mkdir(f"training{training_index}")
    checkpoint_path = f"training{training_index}/cp_{training_index}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
except:
    checkpoint_path = f"training{training_index}/cp_{training_index}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

torch.save(net.state_dict(), checkpoint_path)

print(type(train_losses[1]),type(train_acc[1]))
    
# Plot loss and accuracy

import matplotlib.pyplot as plt
plt.figure()
plt.plot(train_losses,label="train")
plt.plot(val_losses,label = "val")
plt.legend()
plt.ylabel('loss')
plt.xlabel('epoch')
plt.savefig(f"loss{training_index}.png")

plt.figure()
plt.plot(train_acc,label="train")
plt.plot(val_acc,label = "val")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.savefig(f"acc{training_index}.png")

planets = 0
eb = 0
other = 0

data_x.cpu()

for i in range(5):
    #print(i*1000, (i+1)*1000)
    try:
        pred_y = net(data_x[i*1000:(i+1)*1000].cuda())
    except:
        pred_y = net(data_x[i*1000:].cuda())

    for i in pred_y:
        if torch.argmax(i) == 0:
            planets = planets+1
        if torch.argmax(i) == 1:
            eb = eb+1
        if torch.argmax(i) == 2:
            other = other+1
    
    
print(planets,eb,other)
