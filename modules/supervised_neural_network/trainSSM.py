#Train semantig segmentation model
import numpy as np
from keras import backend as K 
import matplotlib.pyplot as plt
from package.CustomDataGenerator import customDataGen
from keras_segmentation.models.unet import vgg_unet

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
train_path = 'Dataset/MOV_0881blue marlin/imagenes etiquetadas'
val_path = 'Dataset/MOV_0881blue marlin/imagenes etiquetadas'

#Get datasets for train and validations
train_generator = customDataGen(train_path, batch_size, input_height, input_width, output_height, output_width, n_classes, EEP_L)
val_generator = customDataGen(val_path, batch_size, input_height, input_width, output_height, output_width, n_classes, EEP_L)

#Create Deep Neural Network Model
model = vgg_unet(n_classes=n_classes ,  input_height=input_height, input_width=input_width, channels=channels)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) # masked_categorical_crossentropy to ignore zero class
history = model.fit(train_generator, epochs=20, validation_data=val_generator)
model.save(model_path) #SAVE MODEL

#End keras and tensorflow session
K.clear_session()
print('TRAINING PROCESS DONE..................................')