
#%% 
import numpy as np

#%% Load the array
data = np.load('dataset/hello/hello_20261022-234210_3c56bb74.npy')

#%% Inspect its structure
print("Shape:", data.shape)
print("Data Type:", data.dtype)

#%% View the actual content
print(data)
