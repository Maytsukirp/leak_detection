#Test semantig segmentation model
import numpy as np
from keras import backend as K 
import matplotlib.pyplot as plt
from keras.models import load_model
from package.imageLoader import ImageLoader
from keras_segmentation.models.unet import vgg_unet
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
images_path = 'Dataset/MOV_0881blue marlin/imagenes etiquetadas'


#Get datasets for train and validations
loader = ImageLoader(images_path, None, None, n_classes, input_height, input_width, output_height, output_width, channels)
images, labels = loader.loadDataSetJson()
#Extended Extintion Profiles
mp = morphologicalProfiles()
images_in = mp.EEP(images, num_levels=EEP_L, mulImgs=True) 

#Load Model Trained
model = load_model(model_path)
print(model.summary())

#Make Predition Using DNN Model
pr = model.predict(np.array([images_in[100,:,:,:]]))[0]
pr = pr.reshape((output_height,  output_width, n_classes)).argmax(axis=2)

gt = labels[0,:,:]
gt = gt.reshape((output_height,  output_width, n_classes)).argmax(axis=2)
#Plot Result
plt.figure()
plt.imshow(pr)
plt.figure()
plt.imshow(gt)
plt.show()

#End keras and tensorflow session
K.clear_session()
print('DONE')

