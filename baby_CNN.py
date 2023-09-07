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

training_index = "baby_CNN_flux_bkg"

tics_of_interest = [230112913,195184329,9697859,288749048,126046118,247019002,349945795,8939649,257219248,
231060598,364393161,9632613,101522288,148039928,416234811,384887159,458195065,72331,152803693, 138667997,435311835,7655219]



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
  
  other_mask = np.where((data_y[:,2]==1))
  other_mask_val = np.where((data_y_val[:,2]==1))
  other_x = data_x[other_mask]
  other_y = data_y[other_mask]
  other_x_val = data_x_val[other_mask_val]
  other_y_val = data_y_val[other_mask_val]
  other_tics = tics_train[other_mask]
  other_tics_val = tics_val[other_mask_val]

  indices = []
  indices_val = []

  for i in tics_of_interest:
     indices.extend(list(np.where(other_tics==i)[0]))
     indices_val.extend(list(np.where(other_tics_val==i)[0]))
  
  #print(indices)
  
  other_x = other_x[indices]
  other_x_val = other_x_val[indices_val]
  other_y = other_y[indices]
  other_y_val = other_y_val[indices_val]
  other_tics = other_tics[indices]
  other_tics_val = other_tics_val[indices_val]


  print(planets_x[::45,:,:].shape,other_x.shape*5)

  data_x = torch.cat((planets_x[::45,:,:],other_x,other_x,other_x,other_x,other_x),0)#nothing_x),0)
  data_y = torch.cat((planets_y[::45,:],other_y,other_y,other_y,other_y, other_y),0)#,nothing_y),0)
  data_x_val = torch.cat((planets_x_val[::45,:,:],other_x_val,other_x_val,other_x_val,other_x_val,other_x_val),0)#,nothing_x_val),0)
  data_y_val = torch.cat((planets_y_val[::45,:],other_y_val,other_y_val,other_y_val,other_y_val,other_y_val),0)#,nothing_y_val),0)
  
  tics_train = torch.cat((planets_tics[::45,:],other_tics,other_tics,other_tics,other_tics,other_tics),0)#,nothing_y),0)
  tics_val = torch.cat((planets_tics_val[::45,:],other_tics_val,other_tics_val,other_tics_val,other_tics_val,other_tics_val),0)#,nothing_x_val),0)
  

  mask = np.array(range(len(data_x)))
  mask_val = np.array(range(len(data_x_val)))

  #print("planets", planets_y.shape, planets_y_val.shape, "fp", other_y.shape, other_y_val.shape, "tics", tics_train.shape, tics_val.shape)#, "nothing", nothing_y.shape, nothing_y_val.shape)
  
  return data_x, data_x_val, data_y[:,[0]], data_y_val[:,[0]], tics_train, tics_val
  
data1x = torch.load("data_x_chunks_1440_no_tics_filterbkg.pt")[:,[0],:]
data2x = torch.load("data_x_chunks_1440_no_tics_filterbkg_2.pt")[:,[0],:]
data3x = torch.load("data_x_chunks_1440_no_tics_filterbkg_3.pt")[:,[0],:]
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
training_generator = torch.utils.data.DataLoader(training_set, batch_size = 16, shuffle=True)

validation_set = torch.utils.data.TensorDataset(data_x_val,data_y_val,tics_val)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = 16, shuffle=True)

print(data_x.shape,data_x_val.shape)

# Build network


channels, n_out = 1,1

class Classifier(nn.Module):
  def __init__(self, channels, n_out):
    super(Classifier,self).__init__()
    self.conv1 = nn.Conv1d(channels, 32, kernel_size=5, padding="same")
    #self.conv2 = nn.Conv1d(32, 64, kernel_size=5, padding="same")
    self.pool1 = nn.MaxPool1d(2)
    self.conv3 = nn.Conv1d(32, 64, kernel_size=5, padding="same")
    #self.conv4 = nn.Conv1d(128,256, kernel_size=5, padding="same")
    #self.conv4 = nn.Conv1d(128,64, kernel_size=5, padding="same")
    self.pool2 = nn.MaxPool1d(2)
    #self.conv5 = nn.Conv1d(256,128, kernel_size=5, padding="same") #new
    #self.conv6 = nn.Conv1d(128,64, kernel_size=5, padding="same") #new
    #self.pool3 = nn.MaxPool1d(2)#new
    self.conv7 = nn.Conv1d(64,32, kernel_size=5, padding="same") #new
    #self.conv8 = nn.Conv1d(32,16, kernel_size=5, padding="same") #new
    self.pool4 = nn.MaxPool1d(2)#new

    self.linear5 = nn.Linear(32, n_out)
    self.dropout = nn.Dropout(0.35) 

  def forward(self, x):
    #print(x.shape)
    x = F.relu(self.conv1(x))
    #x = F.relu(self.conv2(x))
    x = self.dropout(x)
    x = self.pool1(x)
    x = F.relu(self.conv3(x))
    #x = F.relu(self.conv4(x))
    x = self.dropout(x)
    x = self.pool2(x)
    #x = F.relu(self.conv5(x)) #new starts here
    #x = F.relu(self.conv6(x))
    #x = self.dropout(x)
    #x = self.pool3(x)
    x = F.relu(self.conv7(x)) 
    #x = F.relu(self.conv8(x))
    x = self.dropout(x)
    x = self.pool4(x)

    x, _ = x.max(dim=-1) 
    x = F.sigmoid(self.linear5(x))
    return x
    
    

net = Classifier(channels, n_out)
net.cuda()

learn_rate = 0.001

wandb.config.lr = learn_rate
#loss_function = nn.CrossEntropyLoss()#weight = weights)
loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learn_rate)#, weight_decay = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.6)
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.005,cycle_momentum=False,step_size_up=50)

train_losses = []
val_losses = []


train_acc = []
val_acc = []


epochs = 300


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
        train_corr_pre = torch.round(pred_y)==local_labels
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
             val_corr_pre = torch.round(pred_y_val)==local_labels
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

# import matplotlib.pyplot as plt
# plt.figure()
# plt.plot(train_losses,label="train")
# plt.plot(val_losses,label = "val")
# plt.legend()
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.savefig(f"loss{training_index}.png")

# plt.figure()
# plt.plot(train_acc,label=f"train (final = {train_acc[-1]})")
# plt.plot(val_acc,label = f"val (final = {val_acc[-1]})")
# plt.legend()
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend()
# plt.savefig(f"acc{training_index}.png")

planets = 0
other = 0
#nothing = 0 


for local_batch, local_labels, _ in training_generator:
  # Transfer to GPU
  local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
  pred_y = net(local_batch)
  #pred_y = net(data_x[i*1000:(i+1)*1000].cuda())
  for i in torch.round(pred_y):
      if i == 1:
        planets = planets+1
      if i == 0:
        other = other+1
     
    
    
print(planets,other)#,nothing)


end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)

y_true = data_y_val.numpy()
net.eval()
with torch.no_grad():
    y_pred = net(data_x_val.cuda())
y_pred = y_pred.cpu().detach().numpy()


from sklearn.metrics import roc_curve

a, b, c = roc_curve(y_true, y_pred)

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

#y_true = np.argmax(y_true,axis=1)
#y_pred = np.argmax(y_pred,axis=1)
#y_pred = torch.round(y_pred)
#category = np.argmax(y_pred[:,:],axis=1)
#y_pred[np.where(category==0)] = [1,0,0,0]
#y_pred[np.where(category==1)] = [0,1,0,0]
#y_pred[np.where(category==2)] = [0,0,1,0]
#y_pred[np.where(category==3)] = [0,0,0,1]  

print(confusion_matrix(y_true, np.round(y_pred)))

wandb.finish()

