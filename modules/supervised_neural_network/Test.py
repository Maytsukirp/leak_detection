#Test file - Some Algorithms are tested for a future addition to the package
import numpy as np
from keras import backend as K 
import matplotlib.pyplot as plt
from keras.models import load_model
from package.imageLoader import ImageLoader
from package.MorphologicalProfiles import morphologicalProfiles


#Define model parameters
EEP_L = 4
channels = 9
n_classes = 4
batch_size = 10
input_width = 608
input_height = 416
output_width = 304
output_height = 208
model_path = 'model/model_00'
images_path = 'Dataset/MOV_0835north wtr tran/imagenes etiquetadas'


#Get datasets for train and validations
loader = ImageLoader(images_path, None, None, n_classes, input_height, input_width, output_height, output_width, channels)
images, labels = loader.loadDataSetJson()


#Extended Extintion Profiles
mp = morphologicalProfiles()
images = mp.EEP(images, num_levels=EEP_L, mulImgs=True) 
print(images.shape)

np.save('MOV_0835north wtr tran.npy', images)