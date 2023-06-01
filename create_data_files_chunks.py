import lightkurve as lk
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import torch 

labels = []

fluxes = []
bkgs = []
times = []

weird_ones = []

#sector = 18

#sectors = [2,3,4,5,6,17,18,19,45,48]
sectors = [2,3,4,5,6,17,18,19,45]
#cut = 12000
#cut = 2994

chunk_size = 1440

for sec in sectors:

    for curve in glob.glob(f"light_curves_s{sec}/*.npy"):
      
        lc = np.load(curve,allow_pickle = True)
    
        chunk_num = int(len(lc[1])/chunk_size)
      
        if len(lc[1])>=10000:

            for i in range(chunk_num):
                start = int(i*(len(lc[1])-1)/chunk_size)
                flux = np.array(lc[1][start:start+chunk_size])
                flux = flux/np.median(flux) - 1
                flux = list(flux)
                bkg = np.array(lc[6][start:start+chunk_size])
                bkg = bkg/np.median(bkg)
                bkg = list(bkg)
                time = lc[0]
                time = [x - time[0] for x in time]
                time = time[start:start+chunk_size]
          
            
                label = lc[7]
            
                try: 
                    for trans_time in lc[8]:
                
                        if lc[7] == [1, 0, 0]:
                    
                            if float(trans_time) > float(time[0]) and float(trans_time) < float(time[-1]):
                                label = lc[7]
                                break
                        
                            else:
                                label = [0, 0, 1]
                        
                except:
                    print("curve ", curve, "sector ", sec)
                    break
                
                fluxes.append(flux)
                bkgs.append(bkg)
                times.append(time)
                labels.append(label)  
                        




#data = np.array([fluxes,bkgs])
#data = np.reshape(data,(len(),2))

#print(data[534])

def make_tensor(fluxes,bkgs, labels):
  #train_images, val_images, train_labels, val_labels = model_selection.train_test_split(all_images,all_labels, random_state=410)
    data_x = torch.zeros((len(fluxes),2,len(fluxes[0])))
    data_y = torch.zeros((len(labels),3))


  #print(data_x.shape,data_y.shape)

    for i in range(data_y.shape[0]):
        data_y[i,:] = torch.tensor(labels[i])

    for i in range(data_x.shape[0]):

        data_x[i,0,:] = torch.tensor(fluxes[i])
        data_x[i,1,:] = torch.tensor(bkgs[i]) 
    
    
    return data_x, data_y
   
   
   
data_x, data_y = make_tensor(fluxes,bkgs,labels) 

plt.figure()
plt.plot(data_x[15,0,:])
plt.title("preFFT 1")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("pre1.png")

plt.figure()
plt.plot(data_x[4000,0,:])
plt.title("preFFT 2")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("pre2.png")

plt.figure()
plt.plot(data_x[41,1,:])
plt.title("bpreFFT 1 bkg")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("bpre1.png")



# Save pre-processed data

data_x.to(torch.float32)


#torch.save(data_x_clean,f"data_x_s{sector}_clean.pt")  
#torch.save(data_y,f"data_y_s{sector}_clean.pt")  

torch.save(data_x,f"data_x_chunks.pt")  
torch.save(data_y,f"data_y_chunks.pt")     

#print(data_x[:,:,1] == torch.tensor(bkgs))
#print(data_x[1,:,0],fluxes[1])

   
    
  
 
 
 
