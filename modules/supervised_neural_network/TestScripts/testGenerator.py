#Test file - Some Algorithms are tested for a future addition to the package
import numpy as np
from keras import backend as K 
import matplotlib.pyplot as plt
from package.imageLoader import ImageLoader
from package.CustomDataGenerator import customDataGen

EEP_L = 4
n_classes = 4
batch_size =10
input_width = 608
input_height = 416
output_width = 304
output_height = 208
JSON_path = 'Dataset/MOV_0835north wtr tran/imagenes etiquetadas'


train_generator = customDataGen(JSON_path, batch_size, input_height, input_width, output_height, output_width, n_classes, EEP_L)

#Plot Result
plt.figure()
plt.imshow(train_generator[0][0][0,:,:,0])

gt = train_generator[0][1][0]
gt = gt.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
plt.figure()
plt.imshow(gt)
plt.show()