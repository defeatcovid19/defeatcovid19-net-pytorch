import sys
import numpy as np
import sklearn
import pandas as pd
import cv2
import torch

print('Python version: {}'. format(sys.version))
print('NumPy version: {}'. format(np.__version__))
print('scikit-learn version: {}'. format(sklearn.__version__))
print('pandas version: {}'. format(pd.__version__))
print('OpenCV version: {}'. format(cv2.__version__)) 
print('Torch version: {}'. format(torch.__version__))
print('Available GPUs: {}'.format(torch.cuda.device_count()))
if torch.cuda.is_available:
    device = torch.device('cuda')
    print('Cuda version: {}'.format(torch.version.cuda))
else:
    device = torch.device('cpu')
print("Torch device: {}".format(device))