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

weird_ones = []

#sector = 18

sectors = [6,17,18,19,45,48]

cut = 12000
#cut = 2994

for sec in sectors:

  for i in glob.glob(f"light_curves_s{sec}/*.npy"):
    
      lc = np.load(i,allow_pickle = True)
    
      if len(lc[1])>=cut:
          flux = lc[1][(len(lc[1])-cut)//2:-(len(lc[1])-cut)//2]
          bkg = lc[6][(len(lc[6])-cut)//2:-(len(lc[6])-cut)//2]
          fluxes.append(flux)
          bkgs.append(bkg)
        
          label = lc[7]
          labels.append(label)
        
      else:
          pass
    




#data = np.array([fluxes,bkgs])
#data = np.reshape(data,(len(),2))

#print(data[534])

def make_tensor(fluxes,bkgs,labels):
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

# Perform FFT


f_data_x = torch.fft.rfft(data_x, dim= 2)



# Use high and low pass filters

def H_step(f,f_0,n):
    H = 1/(1+(f/f_0)**(2*n))
    return H

#time_spacing = 0.0013888833138935297


frequency = torch.fft.rfftfreq(data_x.shape[2])#,time_spacing)


f_data_x[:,0,:] = f_data_x[:,0,:]*(1-H_step(frequency,0.0001,8))
f_data_x[:,0,:] = f_data_x[:,0,:]*(H_step(frequency,0.47,8))

f_data_x[:,1,:] = f_data_x[:,1,:]*(1-H_step(frequency,0.0001,8))
f_data_x[:,1,:] = f_data_x[:,1,:]*(H_step(frequency,0.47,8))



data_x_clean = torch.real(torch.fft.ifft(f_data_x,dim=2))
#data_x_clean = torch.fft.ifft(data_x_clean,dim=-2)


# Plot some light curves after FFT processing


plt.figure()
plt.plot(torch.real(data_x_clean[15,0,:]))
plt.title("FFT 1")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("fft1.png")
plt.figure()
plt.plot(data_x[15,0,:])
plt.title("preFFT 1")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("pre1.png")

plt.figure()
plt.plot(torch.real(data_x_clean[4000,0,:]))
plt.title("FFT 2")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("fft2.png")
plt.figure()
plt.plot(data_x[4000,0,:])
plt.title("preFFT 2")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("pre2.png")


plt.figure()
plt.plot(data_x_clean[4122,1,:])
plt.title("FFT 1 bkg")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("bfft1.png")
plt.figure()
plt.plot(data_x[4122,1,:])
plt.title("bpreFFT 1 bkg")
plt.xlabel("Time")
plt.ylabel("Flux")
plt.savefig("bpre1.png")



# Save pre-processed data

data_x_clean.to(torch.float32)


#torch.save(data_x_clean,f"data_x_s{sector}_clean.pt")  
#torch.save(data_y,f"data_y_s{sector}_clean.pt")  

torch.save(data_x_clean,f"data_x_6-17-18-19-45-48.pt")  
torch.save(data_y,f"data_y_6-17-18-19-45-48.pt")     

#print(data_x[:,:,1] == torch.tensor(bkgs))
#print(data_x[1,:,0],fluxes[1])

   
    
  
 
 
 
