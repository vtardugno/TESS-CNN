import lightkurve as lk
import os, sys
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import torch  
import scipy as sp

labels = []

fluxes = []
bkgs = []
times = []
mom1s = []
mom2s = []
pos1s = []
pos2s = []
tics = []
fluxes2 = []
sector = []


#sectors = [18,32,33,48]
#sectors = [2,3,5,6,17,19,45]# chunk size without index
#sectors = [4,18,10,11,34] #chunk_size_2
#sectors = [35,36,37,38,42] #chunk_size_3 
#sectors = [50,53,54]

#sectors = [36,37,38,39,40,41,42] #val
#sectors = [43,44,45,46,47,48] #test 1
#sectors = [49,50,51,52,53,54] #test 2
sectors = [55,56,57,58,59,60] #test 2

chunk_size = 1440
stride = 800

for sec in sectors:
    print(sec)
    for curve in glob.glob(f"light_curves_s{sec}/*.npy"):
      
        lc = np.load(curve,allow_pickle = True)
    
        #chunk_num = int(len(lc[1])/chunk_size)
        chunk_num = int(np.ceil((len(lc[1]) - chunk_size) / stride))

        if len(lc[1])>=10000:
            tic = int(curve.split("/")[-1][:-7])

            median_filter = sp.signal.medfilt(lc[6], kernel_size=719)

            for i in range(chunk_num):
                #start = int(i*(len(lc[1])-1)/chunk_size)
                start = i*stride
                flux = np.array(lc[1])/np.median(lc[1]) - 1
                flux = flux[start:start+chunk_size]
                flux = list(flux)

                bkg = np.array(lc[6])/median_filter #divide by median filter
                bkg = bkg[start:start+chunk_size]
                bkg = list(bkg)

                time = lc[0]
                time = [x - time[0] for x in time]
                time = time[start:start+chunk_size]
                
                mom1 = np.array(lc[2])/np.median(lc[2])
                mom1 = mom1[start:start+chunk_size]
                mom1 = list(mom1)

                mom2 = np.array(lc[3])/np.median(lc[3])
                mom2 = mom2[start:start+chunk_size]
                mom2 = list(mom2)

                
                pos1 = np.array(lc[4])/np.median(lc[4])
                pos1 = pos1[start:start+chunk_size]
                pos1 = list(pos1)

                pos2 = np.array(lc[5])/np.median(lc[5])
                pos2 = pos2[start:start+chunk_size]
                pos2 = list(pos2)

                #label = [lc[7][0],lc[7][1],lc[7][2]]
                #flux2 = np.array(lc[1][start:start+chunk_size])
                #flux2 = flux/np.std(flux2) 
                #flux2 = list(flux2)

                label = [lc[7][0],lc[7][1],lc[7][2], 0 ]
            
                try: 
                    for trans_time in lc[8]:
                
                        if lc[7] == [1, 0, 0]:
                    
                            if float(trans_time) > float(time[0]) and float(trans_time) < float(time[-1]):
                                label = [lc[7][0],lc[7][1],lc[7][2], 0 ]
                                break
                        
                            else:
                                label = [0, 0, 0, 1] #"nothing" category
                        
                except:
                    print("curve ", curve, "sector ", sec)
                    break

                try: 
                    for trans_time in lc[8]:
                
                        if lc[7] == [0, 0, 1]:
                    
                            if float(trans_time) > float(time[0]) and float(trans_time) < float(time[-1]):
                                label = [lc[7][0],lc[7][1],lc[7][2], 0 ]
                                break
                        
                            else:
                                label = [0, 0, 0, 1] #"nothing" category
                        
                except:
                    print("curve ", curve, "sector ", sec)
                    break               


                fluxes.append(flux)
                bkgs.append(bkg)
                #times.append(time)
                labels.append(label)  
                mom1s.append(mom1)
                mom2s.append(mom2)
                pos1s.append(pos1)
                pos2s.append(pos2)
                tics.append(tic)
                sector.append(sec)
                #fluxes2.append(flux2)



#print(data[534])

def make_tensor(fluxes,bkgs, labels, mom1s,mom2s,pos1s,pos2s,tics,sector):#,fluxes2):
  #train_images, val_images, train_labels, val_labels = model_selection.train_test_split(all_images,all_labels, random_state=410)
    data_x = torch.zeros((len(fluxes),6,len(fluxes[0])))
    data_y = torch.zeros((len(labels),4))
    tic_ids = torch.zeros((len(tics),1),dtype=torch.int64)
    secs = torch.zeros((len(sector),1),dtype=torch.int64)

  #print(data_x.shape,data_y.shape)

    for i in range(data_y.shape[0]):
        data_y[i,:] = torch.tensor(labels[i])

    #data_y = torch.tensor(labels) #kaze

    for i in range(data_x.shape[0]):

        data_x[i,0,:] = torch.tensor(fluxes[i])
        data_x[i,1,:] = torch.tensor(bkgs[i])
        data_x[i,2,:] = torch.tensor(mom1s[i])
        data_x[i,3,:] = torch.tensor(mom2s[i])
        data_x[i,4,:] = torch.tensor(pos1s[i])
        data_x[i,5,:] = torch.tensor(pos2s[i])
    #     #data_x[i,6,:] = torch.tensor(tics[i]) #remove later
        #data_x[i,6,:] = torch.tensor(fluxes2[i])
    
    #data_x = torch.tensor([fluxes,bkgs,mom1s,mom2s,pos1s,pos2s,fluxes2]).swapaxes(0,1) #kaze

    for i in range(tic_ids.shape[0]):
        tic_ids[i,0] = torch.tensor(tics[i])
    
    for i in range(secs.shape[0]):
        secs[i,0] = torch.tensor(sector[i])
    
    #tic_ids = torch.tensor(tics,dtype=torch.int64) #kaze

    return data_x, data_y, tic_ids, secs
    #return secs
   
   
#data_x, data_y, tic_ids, sector = make_tensor(fluxes,bkgs,labels,mom1s,mom2s,pos1s,pos2s,tics,sector)#),fluxes2) 
test_tensor = make_tensor(fluxes,bkgs,labels,mom1s,mom2s,pos1s,pos2s,tics,sector)#),fluxes2) 

#train_val_tensor = make_tensor(fluxes,bkgs,labels,mom1s,mom2s,pos1s,pos2s,tics,sector)#),fluxes2) 


# Save pre-processed data

#data_x.to(torch.float32)
  

# torch.save(data_x,f"data_x_final.pt")  # Rename these tensors if you'd like
# torch.save(data_y,f"data_y_final.pt")    
# torch.save(tic_ids,f"tic_ids_final.pt")
#torch.save(test_tensor,f"test_tensor_s50_s53_s54.pt")     
   
torch.save(test_tensor,f"test_tensor_3.pt")     


# print(data_y.shape)
#print(data_x.shape)
#print(tic_ids.shape)

# tshirt_pc = np.where((data_y[:,0]==1))[0][:81]
 
# def make_plot(cols, rows, size, array,title,figname,seed=410):
#     np.random.seed(seed)
#     fig, axs = plt.subplots(cols,rows, figsize=size)
#     fig.tight_layout(pad=-1.5)
#     fig.suptitle(title, fontsize=20,y=0.99)
#     plt.subplots_adjust(top=0.96,right=0.99)

#     for row in range(rows):
        
#         for col in range(cols):
            
#             index = np.random.choice(array,replace=False)
            
#             axs[row][col].plot(data_x[index,0,:],".",markersize=0.6,color="indigo",label=tic_ids[index,0])
#             axs[row][col].tick_params(left=False,bottom=False,labelleft=False,labelbottom=False)
#             axs[row][col].legend()
#             array = np.delete(array,np.where(array == index))
#             if col == 6 and row == 7:
#                 print(index)
#             if col == 8 and row == 7:
#                 print(index)
#             if col == 1 and row == 8:
#                 print(index)
#             #array = np.delete(array,np.where(array==index))
#             #axs[row][col].axis("off")
     
#     plt.savefig(figname)    
#     print(len(array))


#make_plot(9, 9, (15,15), tshirt_pc ,"t-shirt plot after tensor creation","tshirt_plot_create_tensor.png")



