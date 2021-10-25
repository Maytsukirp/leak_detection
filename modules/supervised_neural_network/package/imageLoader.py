#Images and annotations loader
import os 
import cv2
import json
import math
import numpy as np
from labelme import utils
import matplotlib.pyplot as plt
from keras_segmentation.data_utils.data_loader import image_segmentation_generator
from keras_segmentation.data_utils.data_loader import verify_segmentation_dataset
from keras_segmentation.data_utils.data_loader import get_segmentation_array
from keras_segmentation.data_utils.data_loader import get_pairs_from_paths
from keras_segmentation.data_utils.data_loader import get_image_array
from keras_segmentation.models.config import IMAGE_ORDERING

class ImageLoader:
    def __init__(self, pathJSON=None, pathImage=None, pathLabel=None, n_classes = 12, input_height=416 , input_width=608, output_height=208, output_width=304, channels=3):
        #Define global variables
        self.pathJSON = pathJSON
        self.pathImage = pathImage
        self.pathLabel = pathLabel
        self.n_classes = n_classes
        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width
        self.channels = channels
        #Dataset verification
        if pathImage != None and pathLabel != None: 
            validate = verify_segmentation_dataset(pathImage, pathLabel, n_classes)

    def loadDataSetImages(self):
        #Get img_seg_pairs of paths to images and labels
        img_seg_pairs = get_pairs_from_paths(self.pathImage, self.pathLabel)
        #Create arrays for images and labels
        images = np.zeros((len(img_seg_pairs), self.input_height, self.input_width, self.channels))
        labels = np.zeros((len(img_seg_pairs), self.output_height*self.output_width, self.n_classes))

        #Loop over img_seg_pairs [0] image - [1] labels
        i = 0
        for pair_set in img_seg_pairs:
            img = cv2.imread(pair_set[0])
            img = get_image_array(img, width=self.input_width, height=self.input_height, ordering=IMAGE_ORDERING)
            images[i,:,:,:] = img #Assing each img to array images
            lbl = cv2.imread(pair_set[1],1)
            lbl = get_segmentation_array(lbl, self.n_classes, self.output_width, self.output_height)
            labels[i,:,:] = lbl #Assing each lbl to array labels
            i += 1

        return images, labels

    def loadDataSetJson(self):
        #list JSON files and read the image in gray scale
        files_list = os.listdir(self.pathJSON) 
        #Create arrays for images and labels
        images = np.zeros((len(files_list), self.input_height, self.input_width))
        labels = np.zeros((len(files_list), self.output_height*self.output_width, self.n_classes))
        #Access to JSON files
        for i in range(len(files_list)):
        #for i in range(len(files_list)):
            path_File = os.path.join(self.pathJSON, files_list[i])
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
        
    def splitDataset(self, images, labels, val_percentage):
        #Divide dataset in training and validation data
        valsection = math.floor(images.shape[0]*val_percentage/100)
        images_train = images[0:(images.shape[0]-valsection),:,:,:]
        labels_train = labels[0:(images.shape[0]-valsection),:,:]
        images_val = images[(images.shape[0]-valsection):,:,:,:]
        labels_val = labels[(images.shape[0]-valsection):,:,:]      
        return images_train, labels_train, images_val, labels_val
    
    def normalizeDataSet(self, images, des_std = True):
        if des_std == True:
            mean = images.mean(axis=0)
            images -= mean
            std = images.std(axis=0)
            images /= std
        if des_std == False:
            images = (images-images.min())/(images.max()-images.min())
        n_images = images
        return n_images

    def graficar_history(self, history):
        if history != 0:
            acc = history.history['accuracy']
            val_acc = history.history['val_acc']
            loss = history.history['loss']
            val_loss = history.history['val_loss']
            plt.figure(1)
            plt.subplot(211)
            epochs = range(1, len(loss) + 1)
            plt.plot(epochs, loss, 'bo', label='Training loss')
            plt.plot(epochs, val_loss, 'b', label='Validation loss')
            plt.title('Training and validation loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()

            plt.subplot(212)
            plt.plot(epochs, acc, 'bo', label='Training acc')
            plt.plot(epochs, val_acc, 'b', label='Validation acc')
            plt.title('Training and validation accuracy')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

    def __str__(self):
        pass