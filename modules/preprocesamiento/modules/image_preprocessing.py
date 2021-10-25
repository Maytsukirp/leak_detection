import cv2 
import numpy as np

class ImageProcessing():
    def __init__(self, image=None):
        if image is None:
            self.image = None
        else:
            self.image = image

    def load_image(self, path):
        self.image = cv2.imread(path)
        return self.image

    def set_image(self, image):
        self.image = image

    def get_image(self):
        return self.image

    def set_false_color(self, name_color):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        if name_color == 'rainbow':
            new_image = cv2.applyColorMap(gray, cv2.COLORMAP_RAINBOW)
        if name_color == 'jet':
            new_image = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        if name_color == 'viridis':
            new_image = cv2.applyColorMap(gray, cv2.COLORMAP_VIRIDIS)
        if name_color == 'magma':
            new_image = cv2.applyColorMap(gray, cv2.COLORMAP_MAGMA)
        self.image = new_image


    def edge_detection(self, threshold1, threshold2):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, threshold1, threshold2)
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        return edges

    def adatative_threshold(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img = cv2.bitwise_not(gray)
        thresh = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
        thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        return thresh

    def contour_detection(self, threshold1, maxval):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, threshold1, maxval, 0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    def contour_drawing(self, contours):
        image = self.image.copy()
        cv2.drawContours(image, contours, -1, (0, 255, 0), 3)
        return image

    # Creat mask for contour_drawing
    def contour_masking(self, contours):
        mask = np.zeros_like(self.image)
        cv2.drawContours(mask, contours, -1, (255, 255, 255), -5)
        return mask

    # Create segmentation from mask 
    def contour_segmentation(self, mask):
        segment = cv2.bitwise_and(self.image, mask)
        return segment

    # Apply erosion to mask the segmentation
    def contour_erosion(self, image, kernel_size, iterations):
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        erosion = cv2.erode(image, kernel, iterations = iterations)
        return erosion

    # Apply dilation to edge detection
    def contour_dilation(self, image, kernel_size, iterations):
        kernel = np.ones((kernel_size, kernel_size),np.uint8)
        dilation = cv2.dilate(image, kernel, iterations = iterations)
        return dilation    
    
    # Segment image from edges
    def contour_segmentation_from_edges(self, edges):
        mask = np.zeros_like(edges)
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask, contours, -1, (255, 25, 20), -1)
        segment = cv2.bitwise_and(self.image, mask)
        return segment

    def gaussian_blur(self, kernel_size):
        kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
        self.image = cv2.filter2D(self.image, -1, kernel)




if __name__ == '__main__':
    image = ImageProcessing()
    image.load_image('../resources/images/image_test.jpg')
    cv2.imshow('original', image.get_image())

    # Edge detection
    edge_detection = image.edge_detection(120,210)
    cv2.imshow('edge_detection', edge_detection)

    # Show contour_drawing
    contours = image.contour_detection(10,100)
    contour_drawing = image.contour_drawing(contours)
    cv2.imshow('contour_drawing', contour_drawing)


    # Show contor_segmentation
    contours = image.contour_detection(10,100)
    #contours = image.contour_filtering(contours)
    mask = image.contour_masking(contours)
    #mask = cv2.bitwise_not(mask)
    cv2.imshow('mask', mask)

    erosion = image.contour_erosion(mask, 5, 8)
    cv2.imshow('erosion', erosion)

    mask = erosion

    #image.set_false_color('rainbow')
    contour_segmentation = image.contour_segmentation(mask)
    cv2.imshow('contour_segmentation', contour_segmentation)


    new_image = ImageProcessing()
    new_image.set_image(edge_detection)
    #new_image.set_false_color('rainbow')
    print(edge_detection.shape)
    edge_contour_segmentation = new_image.contour_segmentation(mask)
    cv2.imshow('edge_contour_segmentation', edge_contour_segmentation)

    new_image_2 = ImageProcessing()
    new_image_2.set_image(edge_contour_segmentation)
    cv2.imshow('new_image_2', new_image_2.get_image())
    
    dilate = new_image_2.contour_dilation(edge_contour_segmentation, 4, 5)
    cv2.imshow('dilate', dilate)
    cv2.imshow('contour_segmentation_2', contour_segmentation)


    image = cv2.bitwise_and(image.get_image(), dilate)
    cv2.imshow('image', image)







    
    cv2.waitKey(0)


    cv2.destroyAllWindows()

    


