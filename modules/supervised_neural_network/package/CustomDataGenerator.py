#Custom Data Generator
import os 
import cv2
import json
import math
import numpy as np
import tensorflow as tf
from labelme import utils
import matplotlib.pyplot as plt
from package.MorphologicalProfiles import morphologicalProfiles
from keras_segmentation.data_utils.data_loader import image_segmentation_generator
from keras_segmentation.data_utils.data_loader import verify_segmentation_dataset
from keras_segmentation.data_utils.data_loader import get_segmentation_array
from keras_segmentation.data_utils.data_loader import get_pairs_from_paths
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models.config import IMAGE_ORDERING


class customDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, path, batch_size, input_height, input_width, output_height, output_width, n_classes, EEP_L, shuffle=True):
        self.path = path
        self.list_IDs = os.listdir(path) 
        self.batch_size = batch_size
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        self.mp = morphologicalProfiles()
        self.EEP_L = EEP_L

    def loadDataSetJson(self, files_list):
        #Create arrays for images and labels
        images = np.zeros((len(files_list), self.input_height, self.input_width))
        labels = np.zeros((len(files_list), self.output_height*self.output_width, self.n_classes))
        #Access to JSON files
        for i in range(len(files_list)):
        #for i in range(len(files_list)):
            path_File = os.path.join(self.path, files_list[i])
            data = json.load(open(path_File))
            img = utils.img_b64_to_arr(data['imageData'])
            imgShape = img.shape
            img = cv2.resize(img, (self.input_width, self.input_height))
            img = img.astype(np.float32)
            img = np.atleast_3d(img)
            img = np.mean(img, axis=2)
            shapes = data['shapes']
            label_name_to_value = {"_background_": 0}
            for shape in shapes:
                label_name = shape["label"]
                if label_name in label_name_to_value:
                    label_value = label_name_to_value[label_name]
                else:
                    label_value = len(label_name_to_value)
                    label_name_to_value[label_name] = label_value
            lbl, lbl_names = utils.shapes_to_label(imgShape, shapes, label_name_to_value)
            seg_labels = np.zeros((self.output_height, self.output_width, self.n_classes))
            lbl = cv2.resize(lbl, (self.output_width, self.output_height), interpolation=cv2.INTER_NEAREST)
            for c in range(self.n_classes):
                seg_labels[:, :, c] = (lbl == c).astype(int)           
            seg_labels = np.reshape(seg_labels, (self.output_width*self.output_height, self.n_classes))            
            images[i,:,:] = img      #Assing each img to array images
            labels[i,:,:] = seg_labels #Assing each lbl to array labels
        return images, labels


    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __getitem__(self, index):
         # Generate indexes of the batch        
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Find list of Elements of Batch
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        #Load data files
        images, y = self.loadDataSetJson(list_IDs_temp)
        #Apply Extended Extinction Profiles
        X = self.mp.EEP(images, num_levels= self.EEP_L, mulImgs=True) 
        return X, y
    
    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))


