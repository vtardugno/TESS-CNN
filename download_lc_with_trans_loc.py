#!/usr/bin/env python
# coding: utf-8

# In[1]:


import lightkurve as lk
import numpy as np
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt


def download_tic(tic, sector):

    # I would check whether the file has already been downloaded first so if the code crashes you can restart it without redownloding everything
    if f"{tic}_lc.txt" not in os.listdir(f"light_curves_s{sector}"):

        try:
            # download data
            search_result = lk.search_lightcurve('TIC' + str(tic), sector = sector, cadence='short', author='SPOC')
            
            lc = search_result.download()#.normalize() #subtract 1 after #divide by median (chunk) sub 1

            time = np.array(lc.time.value)
            flux = np.array(lc.flux.value)
            mom1 = np.array(lc.mom_centr1.value)
            mom2 = np.array(lc.mom_centr2.value)
            pos1 = np.array(lc.pos_corr1.value)
            pos2 = np.array(lc.pos_corr2.value)
            bkg = np.array(lc.sap_bkg.value) #divide by median (chunk) sub 1
            
            

            # make a mask to make sure data is finite
            mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(mom1) & np.isfinite(mom2) & np.isfinite(pos1) & np.isfinite(pos2) & np.isfinite(bkg)

            time = time[mask]
            flux = flux[mask]
            mom1 = mom1[mask]
            mom2 = mom2[mask]
            pos1 = pos1[mask]
            pos2 = pos2[mask]
            bkg  = bkg[mask]

            if (tic in loc_id):
    
                ind = loc_id.index(tic)
                loc = trans_loc.db_peak[ind][1:-1]
                loc = [float(x) for x in loc.split(', ')]
                loc = np.array(loc)
            
                # get the ground truth and make some extra columns
                
                ground_truth_tic = ground_truth[ground_truth['TIC_ID'] == tic]
                ind = ground_truth_tic["final_score"].index[0]


                if ground_truth_tic['final_score'][ind] == 'planet':
                    gt = [1,0,0]
                elif ground_truth_tic['final_score'][ind] == 'EB':
                    gt = [0,1,0]
                else:
                    gt = [0,0,1]

                data = np.array([time,flux, mom1, mom2, pos1, pos2, bkg, gt,loc])

                np.save(f"light_curves_s{sector}/{tic}_lc", data)
                
            
                
            

        except:

            print ("TIC {} failed to download".format(tic))

# run for all of the TIC IDs in the file that I sent 



# In[ ]:


#sectors = [2,3,4,5,6,17,18,19,45,48]
#sectors = [6,17,18,19,45,48]

sectors = [33,48,49]

for sector in sectors:
    # list of TIC IDs + ground truth
    ground_truth = pd.read_csv(f"ground_truth/ground_truth/ground_truth_{sector}.csv")

    tic_ids = ground_truth["TIC_ID"]

    trans_loc = pd.read_csv(f"transit_locs_sec_{sector}.csv")
    loc_id = []

    for i in trans_loc.TIC_ID:
        loc_id.append(i)

    for tic in list(tic_ids):
    
        download_tic(tic,sector)


# In[ ]:




