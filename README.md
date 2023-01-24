# TESS-CNN

create_data_files.py uses light_curve .npy files to create and save data_x and data_y tensors. The data_x tensor contains the flux and background after FFT, and the data_y tensor contains the labels.

TESS_CNN.py contains the convolutional neural network  
