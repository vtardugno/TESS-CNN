# TESS-CNN

downloading_lc.py takes as inputs a sector and a ground_truth_{sector}.csv file and outputs light curve .npy files.

download_lc_with_trans_loc.py takes as inputs a sector, a ground_truth_{sector}.csv fil, and a transit_loc_sec_{sector}.csv file and outputs light curve .npy files.

create_data_files.py uses light_curve .npy files to create and save data_x and data_y tensors. The data_x tensor contains the flux and background, and the data_y tensor contains the labels.

create_data_files_chunks.py uses light_curve .npy files to create and save data_x and data_y tensors. The data_x tensor contains the flux, background, mom1, mom2, pos1, pos2, and TIC ID, and the data_y tensor contains the labels.

TESS_CNN.py contains the first trial convolutional neural network.
  
TESS_CNN_dataloader.py contains the final convolutional neural network.

load_model.py loads the model trained by TESS_CNN_dataloader.py and assesses performance.

Most up-to-date pipeline: 

1) Download light curves with download_lc_with_trans_loc.py
2) Create files with create_data_files_chunks.py
3) Train network with TESS_CNN_dataloader.py (skip this step if you wish to use an existing training run)
4) Load model using load_model.py
