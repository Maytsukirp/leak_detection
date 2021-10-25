import numpy as np
import cv2
from modules.load_image import LoadImage
from modules.image_preprocessing import ImageProcessing

# Create class
class ImagesProcessing():
    def __init__(self):
        self.load_image = LoadImage()
        

    def list_images(self, path_files):
        json_files, path_files = self.load_image.list_json_files(path_files)
        images = self.load_image.list_images_from_json(json_files, path_files)
        return images

    def show_images(self, list_images):
        for image in list_images:
            cv2.imshow('image', image)
            cv2.waitKey(50)        
        cv2.destroyAllWindows()

    def images_processing(self, list_images):
        for image_item in list_images:
            cv2.imshow('original', image_item)
            #kernel = np.array([[0, -1, -1], [1, 0, -1], [1, 1, 0]])
            #image = cv2.filter2D(image_item, -1, kernel)
            #cv2.imshow('sharpen', image)

            image = ImageProcessing(image_item)
            image.gaussian_blur(2)
            # Edge detection
            edge_detection = image.edge_detection(0,255)
            cv2.imshow('edge_detection', edge_detection)

            # Show contour_drawing
            contours = image.contour_detection(10,100)
            contour_drawing = image.contour_drawing(contours)
            cv2.imshow('contour_drawing', contour_drawing)


            # Show contor_segmentation
            contours = image.contour_detection(10,100)
            mask = image.contour_masking(contours)
            #mask = cv2.bitwise_not(mask)
            cv2.imshow('mask', mask)

            erosion = image.contour_erosion(mask, 5, 5)
            #erosion = image.contour_erosion(mask, 1, 1)
            cv2.imshow('erosion', erosion)

            mask = erosion
            #image.set_false_color('rainbow')
            contour_segmentation = image.contour_segmentation(mask)
            cv2.imshow('contour_segmentation', contour_segmentation)


            new_image = ImageProcessing()
            new_image.set_image(edge_detection)
            #new_image.set_false_color('rainbow')
            edge_contour_segmentation = new_image.contour_segmentation(mask)
            cv2.imshow('edge_contour_segmentation', edge_contour_segmentation)

            new_image_2 = ImageProcessing()
            #new_image_2.set_image(edge_contour_segmentation)
            new_image_2.set_image(edge_detection)
            
            dilate = new_image_2.contour_dilation(edge_contour_segmentation, 3, 1)
            #dilate = new_image_2.contour_dilation(edge_detection, 3, 1)
            cv2.imshow('dilate', dilate)


            image_copy  = image.get_image().copy()
            image.set_false_color('rainbow')
            image = cv2.bitwise_and(image.get_image(), dilate)
            
            # add two image 
            image = cv2.add(image, image_copy)

            cv2.imshow('image', image)
            cv2.waitKey(100)        
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def image_preprocessing_2(self, list_images):
        for image_item in list_images:
            image = cv2.convertScaleAbs(image_item, beta=-90)
            cv2.imshow('convertScaleAbs', image)
            image = ImageProcessing(image)
            image.gaussian_blur(2)

            # Show contour_segmentation
            contours = image.contour_detection(25, 100)
            mask = image.contour_masking(contours)
            mask = cv2.bitwise_not(mask)
            cv2.imshow('mask', mask)

            # Show edge_detection
            image_edge = ImageProcessing(mask)
            edge_detection = image_edge.edge_detection(0,255)
            cv2.imshow('edge_detection', edge_detection)
            
            erosion = image_edge.contour_erosion(mask, 2, 5)
            cv2.imshow('erosion', erosion)


            # get contours from image_erosion
            image_countour = ImageProcessing(erosion)
            countours = image_countour.contour_detection(30,100)
            image_countour = image_countour.contour_drawing(countours)
            cv2.imshow('image_countour', image_countour)


            #kernel = np.ones((3,3), np.uint8)
            #opening = cv2.morphologyEx(erosion, cv2.MORPH_OPEN, kernel)
            image_copy  = image.get_image().copy()
            image.set_false_color('rainbow')
            image_erosion = cv2.bitwise_and(image.get_image(), erosion)
            # add two image 
            image_erosion = cv2.add(image_erosion, image_copy)
            cv2.imshow('image_erosion', image_erosion)



            dilate = image_edge.contour_dilation(erosion, 9, 9)
            cv2.imshow('dilate', dilate)
            image.set_false_color('rainbow')
            image_dilate = cv2.bitwise_and(image.get_image(), dilate)
            # add two image 
            image_dilate = cv2.add(image_dilate, image_copy)
            cv2.imshow('image_dilate', image_dilate)

            cv2.waitKey(20)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    images_processing = ImagesProcessing()
    images = images_processing.list_images('../resources/json/label-images')
    images_processing.show_images(images)

        




        
