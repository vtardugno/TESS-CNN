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
mom1s = []
mom2s = []
pos1s = []
pos2s = []

weird_ones = []

#sector = 18

#sectors = [2,3,4,5,6,17,18,19,45,48]
#sectors = [2,3,4,5,6,17,18,19,45]
#sectors = [2,3,5,6,17,19,45]# chunk size without index
#sectors = [4,18,10,11,34] #chunk_size_2
sectors = [35,36,37,38,42] #chunk_size_3 (all of these for when not including moms)

#sectors = [2,3,5]# chunk size without index
#sectors = [4,18,10] #chunk_size_2
#sectors = [35,36,37] #chunk_size_3
#sectors = [19,45,34] #chunk_size_4
#sectors = [17,11,38] #chunk_size_5
#sectors = [6,42] #chunk_size_6

#cut = 12000
#cut = 2994

chunk_size = 1440
#chunk_size = 2880
stride = 800

for sec in sectors:
    print(sec)
    for curve in glob.glob(f"light_curves_s{sec}/*.npy"):
      
        lc = np.load(curve,allow_pickle = True)
    
        #chunk_num = int(len(lc[1])/chunk_size)
        chunk_num = int(np.ceil((len(lc[1]) - chunk_size) / stride))

        if len(lc[1])>=10000:

            for i in range(chunk_num):
                #start = int(i*(len(lc[1])-1)/chunk_size)
                start = i*stride
                flux = np.array(lc[1][start:start+chunk_size])
                flux = flux/np.median(flux) - 1
                flux = list(flux)
                bkg = np.array(lc[6][start:start+chunk_size])
                bkg = bkg/np.median(bkg)
                bkg = list(bkg)
                time = lc[0]
                time = [x - time[0] for x in time]
                time = time[start:start+chunk_size]
                mom1 = np.array(lc[2][start:start+chunk_size])
                mom1 = mom1/np.median(mom1)
                mom1 = list(mom1)
                mom2 = np.array(lc[3][start:start+chunk_size])
                mom2 = mom2/np.median(mom2)
                mom2 = list(mom2)
                pos1 = np.array(lc[4][start:start+chunk_size])
                pos1 = pos1/np.median(pos1)
                pos1 = list(pos1)
                pos2 = np.array(lc[5][start:start+chunk_size])
                pos2 = pos2/np.median(pos2)
                pos2 = list(pos2)
            
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
                times.append(time)
                labels.append(label)  
                mom1s.append(mom1)
                mom2s.append(mom2)
                pos1s.append(pos1)
                pos2s.append(pos2)
                        




#data = np.array([fluxes,bkgs])
#data = np.reshape(data,(len(),2))

#print(data[534])

def make_tensor(fluxes,bkgs, labels, mom1s,mom2s,pos1,pos2):
  #train_images, val_images, train_labels, val_labels = model_selection.train_test_split(all_images,all_labels, random_state=410)
    data_x = torch.zeros((len(fluxes),6,len(fluxes[0])))
    data_y = torch.zeros((len(labels),4))


  #print(data_x.shape,data_y.shape)

    for i in range(data_y.shape[0]):
        data_y[i,:] = torch.tensor(labels[i])

    for i in range(data_x.shape[0]):

        data_x[i,0,:] = torch.tensor(fluxes[i])
        data_x[i,1,:] = torch.tensor(bkgs[i])
        data_x[i,2,:] = torch.tensor(mom1s[i])
        data_x[i,3,:] = torch.tensor(mom2s[i])
        data_x[i,4,:] = torch.tensor(pos1s[i])
        data_x[i,5,:] = torch.tensor(pos2s[i])
    
    
    return data_x, data_y
   
   
   
data_x, data_y = make_tensor(fluxes,bkgs,labels,mom1s,mom2s,pos1s,pos2s) 

plt.figure()
plt.plot(data_x[15,0,:])
plt.title("preFFT 1")
plt.xlabel("Time")
plt.ylabel("Flux")
#plt.savefig("pre1.png")

plt.figure()
plt.plot(data_x[4000,0,:])
plt.title("preFFT 2")
plt.xlabel("Time")
plt.ylabel("Flux")
#plt.savefig("pre2.png")

plt.figure()
plt.plot(data_x[41,1,:])
plt.title("bpreFFT 1 bkg")
plt.xlabel("Time")
plt.ylabel("Flux")
#plt.savefig("bpre1.png")



# Save pre-processed data

data_x.to(torch.float32)


#torch.save(data_x_clean,f"data_x_s{sector}_clean.pt")  
#torch.save(data_y,f"data_y_s{sector}_clean.pt")  

torch.save(data_x,f"data_x_chunks_{chunk_size}_3.pt")  
torch.save(data_y,f"data_y_chunks_{chunk_size}_3.pt")     

#print(data_x[:,:,1] == torch.tensor(bkgs))
#print(data_x[1,:,0],fluxes[1])

   
print(data_y.shape)
print(data_y[:5])
print(data_x.shape)
  
 
 
 
