# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch.nn.functional as F
import os 
import torch
import torch.nn as nn
import time
import wandb


start_time = time.time()

torch.autograd.set_detect_anomaly(True)
np.random.seed(410)

training_index = "1440chunks_21_bkg_only"


wandb.init(project="TESS_CNN_dataloader",name=training_index)

def prep_data(all_x,all_y,split,tics):
  #all_x = torch.load(x)
  #all_y = torch.load(y)

  #all_x = torch.reshape(all_x, (all_x.shape[0],all_x.shape[2],all_x.shape[1]))

  data_x = all_x[:split,:,:]
  data_x_val = all_x[split:,:,:]
  data_y = all_y[:split,:]
  data_y_val = all_y[split:,:]

  tics_train = tics[:split]
  tics_val = tics[split:]

  planet_mask = np.where((data_y[:,0]==1))
  planet_mask_val = np.where((data_y_val[:,0]==1))
  planets_x = data_x[planet_mask]
  planets_y = data_y[planet_mask]
  planets_x_val = data_x_val[planet_mask_val]
  planets_y_val = data_y_val[planet_mask_val]
  planets_tics = tics_train[planet_mask]
  planets_tics_val = tics_val[planet_mask_val]
  
  eb_mask = np.where((data_y[:,1]==1))
  eb_mask_val = np.where((data_y_val[:,1]==1))
  eb_x = data_x[eb_mask]
  eb_y = data_y[eb_mask]
  eb_x_val = data_x_val[eb_mask_val]
  eb_y_val = data_y_val[eb_mask_val]
  eb_tics = tics_train[eb_mask]
  eb_tics_val = tics_val[eb_mask_val]

  other_mask = np.where((data_y[:,2]==1))
  other_mask_val = np.where((data_y_val[:,2]==1))
  other_x = data_x[other_mask]
  other_y = data_y[other_mask]
  other_x_val = data_x_val[other_mask_val]
  other_y_val = data_y_val[other_mask_val]
  other_tics = tics_train[other_mask]
  other_tics_val = tics_val[other_mask_val]

  data_x = torch.cat((planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,eb_x,other_x,other_x,other_x,other_x),0)#nothing_x),0)
  data_y = torch.cat((planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,eb_y,other_y,other_y,other_y,other_y),0)#,nothing_y),0)
  data_x_val = torch.cat((planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,eb_x_val,other_x_val,other_x_val,other_x_val,other_x_val),0)#,nothing_x_val),0)
  data_y_val = torch.cat((planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,eb_y_val,other_y_val,other_y_val,other_y_val,other_y_val),0)#,nothing_y_val),0)
  
  tics_train = torch.cat((planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,eb_tics,other_tics,other_tics,other_tics,other_tics),0)#,nothing_y),0)
  tics_val = torch.cat((planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,eb_tics_val,other_tics_val,other_tics_val,other_tics_val,other_tics_val),0)#,nothing_x_val),0)
  



  mask = np.array(range(len(data_x)))
  mask_val = np.array(range(len(data_x_val)))

  #np.random.shuffle(mask)
  #np.random.shuffle(mask_val)


  #data_x = data_x[mask]
  #data_y = data_y[mask]
  #data_x_val = data_x_val[mask_val]
  #data_y_val = data_y_val[mask_val]
  
 
  print("planets", planets_y.shape, planets_y_val.shape, "eb", eb_y.shape, eb_y_val.shape, "fp", other_y.shape, other_y_val.shape, "tics", tics_train.shape, tics_val.shape)#, "nothing", nothing_y.shape, nothing_y_val.shape)
  
  return data_x, data_x_val, data_y[:,:-1], data_y_val[:,:-1], tics_train, tics_val
  
data1x = torch.load("data_x_chunks_1440_no_tics_filterbkg.pt")[:,[1],:]
data2x = torch.load("data_x_chunks_1440_no_tics_filterbkg_2.pt")[:,[1],:]
data3x = torch.load("data_x_chunks_1440_no_tics_filterbkg_3.pt")[:,[1],:]
#data4x = torch.load("data_x_chunks_1440_4.pt")
#data5x = torch.load("data_x_chunks_1440_5.pt")
#data6x = torch.load("data_x_chunks_1440_6.pt")

data1y = torch.load("data_y_chunks_1440_no_tics_filterbkg.pt")
data2y = torch.load("data_y_chunks_1440_no_tics_filterbkg_2.pt")
data3y = torch.load("data_y_chunks_1440_no_tics_filterbkg_3.pt")
#data4y = torch.load("data_y_chunks_1440_4.pt")
#data5y = torch.load("data_y_chunks_1440_5.pt")
#data6y = torch.load("data_y_chunks_1440_6.pt")

id1 = torch.load("tic_ids_chunks_1440_no_tics_filterbkg.pt")
id2 = torch.load("tic_ids_chunks_1440_no_tics_filterbkg_2.pt")
id3 = torch.load("tic_ids_chunks_1440_no_tics_filterbkg_3.pt")


datax = torch.cat((data1x,data2x,data3x),0)
datay = torch.cat((data1y,data2y,data3y),0)
ids = torch.cat((id1,id2,id3),0)



print(datax.shape,datay.shape,ids.shape)

mask = np.array(range(len(datax)))
np.random.shuffle(mask)
datax = datax[mask]
datay = datay[mask]
ids = ids[mask]


data_x, data_x_val, data_y, data_y_val, tics_train, tics_val = prep_data(datax, datay ,225000,ids)


mask = np.array(range(len(data_x)))
mask_val = np.array(range(len(data_x_val)))

np.random.shuffle(mask)
np.random.shuffle(mask_val)



data_x = data_x[mask]
data_y = data_y[mask]
data_x_val = data_x_val[mask_val]
data_y_val = data_y_val[mask_val]

tics_train = tics_train[mask]
tics_val = tics_val[mask_val]

#print(data_x.shape,data_x_val.shape,data_y.shape,data_y_val.shape,tics_train.shape,tics_val.train)

training_set = torch.utils.data.TensorDataset(data_x,data_y, tics_train)
training_generator = torch.utils.data.DataLoader(training_set, batch_size = 1024, shuffle=True)

validation_set = torch.utils.data.TensorDataset(data_x_val,data_y_val,tics_val)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = 1024, shuffle=True)

print(data_x.shape,data_x_val.shape)

# Build network


channels, n_out = 1,3

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
net.cuda()

# Give more weight to the planet candidates when calculating the loss

#neg = torch.sum(data_y[:,2]!=1)
#pos = torch.sum(data_y[:,2]==1)
#total = neg+pos

#w_0 = (1/neg)*(total/2)
#w_1 = (1/pos)*(total/2)

#weights = [w_1,w_0,w_0,w_0]
#weights = torch.tensor(weights).cuda()

loss_function = nn.CrossEntropyLoss()#weight = weights)
#loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=0.001)#, weight_decay = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.8)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.005,cycle_momentum=False,step_size_up=50)

train_losses = []
val_losses = []


train_acc = []
val_acc = []


epochs = 200


# Train the network

for epoch in range(epochs):

    print(f"epoch: {epoch}")
    
    net.eval()
    
    loss_loop = []
    val_loss_loop = []
    acc_loop = []
    val_acc_loop  = []

    for local_batch, local_labels, _ in training_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
        net.train()
        pred_y = net(local_batch)
        loss = loss_function(pred_y, local_labels)
        loss_loop.append(loss.item())
        train_corr_pre = torch.argmax(pred_y,dim=1)==torch.argmax(local_labels,dim=1)
        #train_corr_pre = torch.round(pred_y) == local_labels
        train_corr = torch.sum(train_corr_pre)
        acc_loop.append((train_corr/len(pred_y)).cpu())
        net.zero_grad()
        loss.backward()
        optimizer.step()
         
    for local_batch, local_labels, _ in validation_generator:
        # Transfer to GPU
        local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
        net.eval()
        with torch.no_grad():
             pred_y_val = net(local_batch)
             val_loss = loss_function(pred_y_val, local_labels)
             val_loss_loop.append(val_loss.item())
             val_corr_pre = torch.argmax(pred_y_val,dim=1)==torch.argmax(local_labels,dim=1)
             #val_corr_pre = torch.round(pred_y_val) == local_labels
             val_corr = torch.sum(val_corr_pre)
             val_acc_loop.append((val_corr/len(pred_y_val)).cpu())


    val_losses.append(np.average(np.array(val_loss_loop)))
    train_losses.append(np.average(np.array(loss_loop)))
        
    train_acc.append(np.average(np.array(acc_loop)))
    val_acc.append(np.average(np.array(val_acc_loop)))

    wandb.log({"train": {"acc": np.average(np.array(acc_loop)),"loss":np.average(np.array(loss_loop))}, "val": {"acc": np.average(np.array(val_acc_loop)),"loss": np.average(np.array(val_loss_loop))}})

    #net.zero_grad()
    #loss.backward()


    scheduler.step()
    

# Create a directory and save the training results. If the directory already exists, add them there

try:
    os.mkdir(f"training{training_index}")
    checkpoint_path = f"training{training_index}/cp_{training_index}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
except:
    checkpoint_path = f"training{training_index}/cp_{training_index}.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

torch.save(net.state_dict(), checkpoint_path)

#print(train_losses)
    
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
plt.plot(train_acc,label=f"train (final = {train_acc[-1]})")
plt.plot(val_acc,label = f"val (final = {val_acc[-1]})")
plt.legend()
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend()
plt.savefig(f"acc{training_index}.png")

planets = 0
eb = 0
other = 0
#nothing = 0 


for local_batch, local_labels, _ in training_generator:
  # Transfer to GPU
  local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
  pred_y = net(local_batch)
  #pred_y = net(data_x[i*1000:(i+1)*1000].cuda())
  for i in pred_y:
      if torch.argmax(i) == 0:
        planets = planets+1
      if torch.argmax(i) == 1:
        eb = eb+1
      if torch.argmax(i) == 2:
        other = other+1
      #if torch.argmax(i) == 3:
       # nothing = nothing+1
      #if torch.round(i) == 0:
       # eb = eb+1
      #else:
       # planets = planets +1
    
    
print(planets,eb,other)#,nothing)


end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)

y_true = data_y_val[::8,:].numpy()
net.eval()
with torch.no_grad():
    y_pred = net(data_x_val[::8,:,:].cuda())
y_pred = y_pred.cpu().detach().numpy()


from sklearn.metrics import roc_curve

a, b, c = roc_curve(y_true[:,0], y_pred[:,0])

plt.figure()
plt.plot(a,b,label="PC vs the rest")
plt.plot([0, 1], [0, 1], "k--", label="chance level (AUC = 0.5)")
plt.axis("square")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("One-vs-Rest ROC curves:\nPC vs the rest")
plt.legend()
plt.savefig(f"ROC_{training_index}.png")

from sklearn.metrics import confusion_matrix

y_true = np.argmax(y_true,axis=1)
y_pred = np.argmax(y_pred,axis=1)
#y_pred = torch.round(y_pred)
#category = np.argmax(y_pred[:,:],axis=1)
#y_pred[np.where(category==0)] = [1,0,0,0]
#y_pred[np.where(category==1)] = [0,1,0,0]
#y_pred[np.where(category==2)] = [0,0,1,0]
#y_pred[np.where(category==3)] = [0,0,0,1]  

print(confusion_matrix(y_true, y_pred))

wandb.finish()
