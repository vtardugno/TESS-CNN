# TESS-CNN

downloading_lc.py takes as inputs a sector and a ground_truth_{sector}.csv file and outputs light curve .npy files

create_data_files.py uses light_curve .npy files to create and save data_x and data_y tensors. The data_x tensor contains the flux and background, and the data_y tensor contains the labels.

TESS_CNN.py contains the first trial convolutional neural network

  
TESS_CNN_dataloader.py contains the final convolutional neural network
