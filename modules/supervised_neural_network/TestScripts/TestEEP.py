#TEST EEP
import numpy as np
from keras import backend as K 
import matplotlib.pyplot as plt
from package.imageLoader import ImageLoader
from package.MorphologicalProfiles import morphologicalProfiles


#Define model parameters
channels = 3
n_classes = 4
input_width = 608
output_width = 304
input_height = 416
output_height = 208
model_path = 'model/model_00'
JSON_path = 'Dataset/MOV_0835north wtr tran/imagenes etiquetadas'

#Get datasets for train and validations
loader = ImageLoader(pathJSON=JSON_path, n_classes=n_classes, input_height=input_height, input_width=input_width, output_height=output_height, output_width=output_width, channels=channels)
images, labels = loader.loadDataSetJson()
#images = loader.normalizeDataSet(images, des_std=True)
print(images.shape)
#Plot Result
plt.figure()
plt.imshow(images[0])
plt.show()

#Extended Extinction Profiles
print(images[0].shape)
mp = morphologicalProfiles()
imagenFE = mp.EEP(images, num_levels=4, mulImgs=True)    #TENSOR (N_img, h, w, EEP)
print(imagenFE.shape)

for i in range(9):
  plt.figure()
  plt.imshow(imagenFE[0,:,:,i])
  plt.show()

#images_train, labels_train, images_val, labels_val = loader.splitDataset(images, labels, val_percentage=25) 