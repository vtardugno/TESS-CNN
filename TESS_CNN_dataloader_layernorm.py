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

training_index = "20"

os.makedirs(f"training{training_index}",exist_ok=True)
checkpoint_path = f"training{training_index}/cp_{training_index}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

best_path = f"training{training_index}/best_{training_index}.ckpt"
best_dir = os.path.dirname(best_path)

learn_rate = 0.0005
epoch_sched = 5
gamma_sched = 0.8
dropout = 0.1
channels, n_out = 6,3
batch = 1024
epochs = 100
val_cut = 207000
#val_cut = 50000

configs = {"lr": learn_rate,
           "epoch_sched": epoch_sched,
           "gamma_sched": gamma_sched,
           "dropout":dropout,
           "out_dim":n_out,
           "in_channels":channels,
           "batch_size": batch,
           "epochs":epochs,
           "val_cut":val_cut,
           "loss_weights":"yes",
           "TOIs":"no",
           "batch_norm": "no, layer norm"}

wandb.init(project="TESS_CNN_dataloader",name=f"{training_index}")
wandb.config.update(configs)
def prep_data(all_x,all_y,split,tics):

  data_x = all_x[:split,:,:]
  data_x_val = all_x[split:,:,:]
  data_y = all_y[:split,:]
  data_y_val = all_y[split:,:]

  tics_train = tics[:split]
  tics_val = tics[split:]

  planet_mask = np.where((data_y[:,0]==1))
  #planet_mask_val = np.where((data_y_val[:,0]==1))
  planets_x = data_x[planet_mask]
  planets_y = data_y[planet_mask]
  # planets_x_val = data_x_val[planet_mask_val]
  # planets_y_val = data_y_val[planet_mask_val]
  planets_tics = tics_train[planet_mask]
  # planets_tics_val = tics_val[planet_mask_val]
  
  eb_mask = np.where((data_y[:,1]==1))
  #eb_mask_val = np.where((data_y_val[:,1]==1))
  eb_x = data_x[eb_mask]
  eb_y = data_y[eb_mask]
  #eb_x_val = data_x_val[eb_mask_val]
  #eb_y_val = data_y_val[eb_mask_val]
  eb_tics = tics_train[eb_mask]
  #eb_tics_val = tics_val[eb_mask_val]

  other_mask = np.where((data_y[:,2]==1))
  #other_mask_val = np.where((data_y_val[:,2]==1))
  other_x = data_x[other_mask]
  other_y = data_y[other_mask]
  #other_x_val = data_x_val[other_mask_val]
  #other_y_val = data_y_val[other_mask_val]
  other_tics = tics_train[other_mask]
  #other_tics_val = tics_val[other_mask_val]

  nothing_mask = np.where((data_y[:,3]==1))
  #nothing_mask_val = np.where((data_y_val[:,3]==1))

  nothing_x = data_x[nothing_mask]
  nothing_y = data_y[nothing_mask]
  #nothing_x_val = data_x_val[nothing_mask_val]
  #nothing_y_val = data_y_val[nothing_mask_val]
  nothing_tics = tics_train[nothing_mask]
  #nothing_tics_val = tics_val[nothing_mask_val]

  data_x = torch.cat((planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,planets_x,eb_x,other_x,other_x,other_x,other_x,nothing_x[::1000,:,:]),0)
  data_y = torch.cat((planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,planets_y,eb_y,other_y,other_y,other_y,other_y,nothing_y[::1000,:]),0)
  #data_x_val = torch.cat((planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,planets_x_val,eb_x_val,other_x_val,other_x_val,other_x_val,other_x_val,nothing_x_val),0)
  #data_y_val = torch.cat((planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,planets_y_val,eb_y_val,other_y_val,other_y_val,other_y_val,other_y_val,nothing_y_val),0)
  
  tics_train = torch.cat((planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,planets_tics,eb_tics,other_tics,other_tics,other_tics,other_tics,nothing_tics[::1000,:]),0)
  #tics_val = torch.cat((planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,planets_tics_val,eb_tics_val,other_tics_val,other_tics_val,other_tics_val,other_tics_val,nothing_tics_val),0)
  
  
 
  print("planets", planets_y.shape, "eb", eb_y.shape, "fp", other_y.shape,  "tics", tics_train.shape, tics_val.shape, "nothing", nothing_y.shape)
  
  return data_x, data_x_val, data_y, data_y_val, tics_train, tics_val
  
# data1x = torch.load("data_x_chunks_1440_no_tics_filterbkg.pt")#[:,[0],:]
# data2x = torch.load("data_x_chunks_1440_no_tics_filterbkg_2.pt")#[:,[0],:]
# data3x = torch.load("data_x_chunks_1440_no_tics_filterbkg.pt")#[:,[0],:]

# data1y = torch.load("data_y_chunks_1440_no_tics_filterbkg.pt")
# data2y = torch.load("data_y_chunks_1440_no_tics_filterbkg_2.pt")
# data3y = torch.load("data_y_chunks_1440_no_tics_filterbkg_3.pt")
# #data4y = torch.load("data_y_chunks_1440_4.pt")
# #data5y = torch.load("data_y_chunks_1440_5.pt")
# #data6y = torch.load("data_y_chunks_1440_6.pt")

# id1 = torch.load("tic_ids_chunks_1440_no_tics_filterbkg.pt")
# id2 = torch.load("tic_ids_chunks_1440_no_tics_filterbkg_2.pt")
# id3 = torch.load("tic_ids_chunks_1440_no_tics_filterbkg_3.pt")

tensor_1 = torch.load("train_val_tensors.pt")
tensor_2 = torch.load("train_val_tensors_2.pt")
tensor_3 = torch.load("train_val_tensors_3.pt")

#print(tensor_2.shape,tensor_3.shape)            
# datax = torch.cat((data1x,data2x,data3x),0)
# datay = torch.cat((data1y,data2y,data3y),0)
# ids = torch.cat((id1,id2,id3),0)

datax = torch.cat((tensor_1[0],tensor_2[0],tensor_3[0]),0)
datay = torch.cat((tensor_1[1],tensor_2[1],tensor_3[1]),0)
ids = torch.cat((tensor_1[2], tensor_2[2],tensor_3[2]),0)

# mask = ~torch.any(datax[:,0,:].isnan(),dim=1)
# datax = datax[mask]
# datay = datay[mask]
# ids = ids[mask]

#print(datax.shape,datay.shape,ids.shape)

# mask = np.array(range(len(datax)))
# np.random.shuffle(mask)
# datax = datax[mask]
# datay = datay[mask]
# ids = ids[mask]


data_x, data_x_val, data_y, data_y_val, tics_train, tics_val = prep_data(datax, datay ,val_cut,ids)

#data_x, data_x_val, data_y, data_y_val, tics_train, tics_val = prep_data(tensor[0], tensor[1] ,val_cut,tensor[2])

tois = pd.read_csv("exofop_tess_tois-2023512.csv")["TIC ID"]
mask_train = np.isin(tics_train,tois,invert=True)
mask_train = np.nonzero(mask_train)[0]

mask_val = np.isin(tics_val,tois,invert=True)
mask_val = np.nonzero(mask_val)[0]

data_x, data_x_val, data_y, data_y_val, tics_train, tics_val = data_x[mask_train,:,:], data_x_val[mask_val,:,:], data_y[mask_train,:], data_y_val[mask_val,:], tics_train[mask_train,:], tics_val[mask_val,:]

#print(data_x.shape,data_x_val.shape,data_y.shape,data_y_val.shape,tics_train.shape,tics_val.train)

training_set = torch.utils.data.TensorDataset(data_x,data_y, tics_train)
training_generator = torch.utils.data.DataLoader(training_set, batch_size = batch, shuffle=True)

validation_set = torch.utils.data.TensorDataset(data_x_val,data_y_val,tics_val)
validation_generator = torch.utils.data.DataLoader(validation_set,batch_size = batch, shuffle=True)

print(data_x.shape,data_x_val.shape)

# Build network


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
        #self.dropout = nn.Dropout(dropout) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.layernorm1(self.conv1_2(x)))
        #x = self.dropout(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.layernorm2(self.conv2_2(x)))
        #x = self.dropout(x)
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.layernorm3(self.conv3_2(x)))
        #x = self.dropout(x)
        x = self.pool3(x)
        x = F.relu(self.conv4(x))
        x = F.relu(self.layernorm4(self.conv4_2(x)))
        #x = self.dropout(x)
        x = self.pool4(x)
        x = F.relu(self.conv5(x))  # new starts here
        x = F.relu(self.layernorm5(self.conv6(x)))
        #x = self.dropout(x)
        x = self.pool5(x)
        x = F.relu(self.conv7(x))
        x = F.relu(self.layernorm6(self.conv8(x)))
        x = self.pool6(x)  # new ends here

        x, _ = x.max(dim=-1)
        x = F.softmax(self.linear5(x), dim=1)
        return x
    

net = Classifier(channels, n_out)
net.cuda()

wandb.watch(net,log="all")

# Give more weight to the planet candidates when calculating the loss

planet_num = torch.sum(data_y[:,0]==1)
eb_num = torch.sum(data_y[:,1]==1)
fp_num = torch.sum(data_y[:,2]==1) + torch.sum(data_y[:,3]==1)
total = planet_num + eb_num + fp_num

def weight(total_samples,num_samples_in_class_i):
  return total_samples / (num_samples_in_class_i * 3)

weights = [weight(total,planet_num),weight(total,eb_num),weight(total,fp_num)]
weights = torch.tensor(weights).cuda()

loss_function = nn.CrossEntropyLoss(weight = weights)
#loss_function = nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(),lr=learn_rate)#, weight_decay = 0.9)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epoch_sched, gamma=gamma_sched) # works: step 60 gamma 0.8
#scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0005, max_lr=0.005,cycle_momentum=False,step_size_up=50)

train_losses = []
val_losses = []

train_acc = []
val_acc = []




best_loss = 1000

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
        labels_new = local_labels[:,:-1].clone()
        labels_new[:,-1] += local_labels[:,-1].clone()
        loss = loss_function(pred_y, labels_new)
        loss_loop.append(loss.item())
        train_corr_pre = torch.argmax(pred_y,dim=1)==torch.argmax(labels_new,dim=1)
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
             labels_new_val = local_labels[:,:-1].clone()
             labels_new_val[:,-1] += local_labels[:,-1].clone()
             #print("pred y nans:", torch.sum(torch.isnan(pred_y_val)))
             #print("labels new nans:", torch.sum(torch.isnan(labels_new_val)))
             val_loss = loss_function(pred_y_val, labels_new_val)
             val_loss_loop.append(val_loss.item())
             #print(val_loss)
             val_corr_pre = torch.argmax(pred_y_val,dim=1)==torch.argmax(labels_new_val,dim=1)
             #val_corr_pre = torch.round(pred_y_val) == local_labels
             val_corr = torch.sum(val_corr_pre)
             val_acc_loop.append((val_corr/len(pred_y_val)).cpu())

    #print(np.average(np.array(val_loss_loop)))
    val_losses.append(np.average(np.array(val_loss_loop)))
    train_losses.append(np.average(np.array(loss_loop)))
        
    train_acc.append(np.average(np.array(acc_loop)))
    val_acc.append(np.average(np.array(val_acc_loop)))

    wandb.log({"train": {"acc": np.average(np.array(acc_loop)),"loss":np.average(np.array(loss_loop))}, "val": {"acc": np.average(np.array(val_acc_loop)),"loss": np.average(np.array(val_loss_loop))}})
    #net.zero_grad()
    #loss.backward()


    scheduler.step()
    torch.save(net.state_dict(), checkpoint_path)
    wandb.save(checkpoint_path)

    if np.average(np.array(val_loss_loop)) < best_loss:
       best_loss = np.average(np.array(val_loss_loop))
       torch.save(net.state_dict(), best_path)
       wandb.save(best_path)
       


# Create a directory and save the training results. If the directory already exists, add them there


#torch.save(net.state_dict(), checkpoint_path)

#print(train_losses)
    
# Plot loss and accuracy

import matplotlib.pyplot as plt
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

# planets = 0
# eb = 0
# other = 0

y_true = np.array([])
y_pred = np.array([])
tic_ids = np.array([])

net.load_state_dict(torch.load(f'training{training_index}/best_{training_index}.ckpt',map_location=torch.device('cpu')))
net.eval()
for local_batch, local_labels, tics in validation_generator:
  # Transfer to GPU
  local_batch, local_labels = local_batch.cuda(), local_labels.cuda()
  pred_y = net(local_batch).cpu().detach().numpy()
  true_y = local_labels.cpu().numpy()
  if y_true.shape[0] == 0:
    y_true = true_y
    y_pred = pred_y
    tic_ids = tics.cpu().numpy()
  else:
    y_true = np.append(y_true, true_y, axis=0)
    y_pred = np.append(y_pred, pred_y, axis=0)
    tic_ids = np.append(tic_ids, tics, axis=0)
  #pred_y = net(data_x[i*1000:(i+1)*1000].cuda())
  # for i in pred_y:
  #     if np.argmax(i) == 0:
  #       planets = planets+1
  #     if np.argmax(i) == 1:
  #       eb = eb+1
  #     if np.argmax(i) == 2:
  #       other = other+1


#print(planets,eb,other)


end_time = time.time()
elapsed_time = end_time - start_time

print("Elapsed time: ", elapsed_time)


np.save(f"y_true_val_{training_index}.npy",y_true)
np.save(f"y_pred_val_{training_index}.npy",y_pred)
np.save(f"y_tic_val_{training_index}.npy",tic_ids)

# df = pd.DataFrame({"y_true":y_true,"y_pred":y_pred,"tic_ids":tic_ids})

# wandb.log(wandb.Table(dataframe=df))

# wandb.log({f"y_true_val_{training_index}.npy":y_true,
#            f"y_pred_val_{training_index}.npy":y_pred,
#            f"y_tic_val_{training_index}.npy":tic_ids})

# from sklearn.metrics import roc_curve

# planet_score = y_pred[:,0]

# current_top_1000 = pd.read_csv(f"top_1000_sec_{sector}.csv")["TIC_ID"]
# current_top_1000_scores = pd.read_csv(f"top_1000_sec_{sector}.csv")["db_count_weighted"]

# sorting_mask = np.flip(np.argsort(planet_score))

# sorted_labels = y_true[sorting_mask,0].numpy()
# sorted_tics = tic_ids[sorting_mask,0].numpy()
# sorted_preds = y_pred[sorting_mask,0] 

# unique_sorted_tics, unique_mask = np.unique(sorted_tics, return_index=True)
# unique_sorted_tics = unique_sorted_tics[np.argsort(unique_mask)]
# unique_sorted_labels = sorted_labels[np.argsort(unique_mask)]
# unique_sorted_preds = sorted_preds[np.argsort(unique_mask)]

# filter_mask = np.isin(unique_sorted_tics,current_top_1000)

# top_500_cnn_tics = unique_sorted_tics[filter_mask]
# top_500_cnn_labels = unique_sorted_labels[filter_mask]
# top_500_cnn_scores = unique_sorted_preds[filter_mask]

# current_top_500_tics = current_top_1000[:500]
# current_top_500_scores = current_top_1000_scores[:500]

# label_mask = np.isin(unique_sorted_tics,current_top_500_tics)

# current_top_500_labels = unique_sorted_labels[label_mask]
# current_top_500_scores[label_mask]

# final_df1 = pd.DataFrame(np.hstack((top_500_cnn_tics,current_top_500_tics)), 
#                   columns=["top_500_cnn_tics","top_500_cnn_scores","top_500_cnn_labels","current_top_500_tics","current_top_500_labels"])

# final_df2 = pd.DataFrame(np.hstack((top_500_cnn_scores,top_500_cnn_labels,current_top_500_scores,current_top_500_labels)), 
#                   columns=["top_500_cnn_tics","top_500_cnn_scores","top_500_cnn_labels","current_top_500_tics","current_top_500_labels"])


# final_df1.to_csv("results_df_tics.csv")
# final_df2.to_csv("results_df_scores.csv")

# wandb.log({"val_results_tics":wandb.Table(dataframe=final_df1),"val_results_scores":wandb.Table(dataframe=final_df2)})

wandb.finish()
