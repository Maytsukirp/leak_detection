import json
import os
from labelme import utils
import cv2

class LoadImage():
    def __init__(self):
        self.image_path = ""
        self.image = None
        self.image_file = None

    def load_json(self, path_file):
        """
        Load json file 
        load_json(path_file) -> json
        parameter:
            :param path_file: path of json file
        return:
            :json_data: json
        """
        with open(path_file, 'r') as f:
            json_data = json.load(f)
        return json_data

    def _string_to_image(self, image_string):
        """
        Convert string image to image
        Parameter:
            :image_string: (string) image string
        Return:
            :image: (array.np) image
        """
        image = utils.img_b64_to_arr(image_string)
        return image

    def image_from_json(self, json_data):
        """
        Load image from json file
        image_from_json(json_data) -> image
        parameter:
            :param json_data: json
        return:
            :image: image
        """
        image_string = json_data['imageData']
        image = self._string_to_image(image_string)
        return image 

    def show_image(self, image):
        """
        Show image with opencv
        show_image(image) -> image
        parameter:
            :param image: image
        """
        cv2.imshow('image', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def save_image(self, image, path_file):
        """
        Save image to file
        save_image(image, path_file) -> image
        parameter:
            :param image: image
            :param path_file: path of image file
        """
        cv2.imwrite(path_file, image)
    
    def list_json_files(self, path_file):
        """
        List all json files in path_file
        list_json_files(path_file) -> json_files
        parameter:
            :param path_file: path of json file
        return:
            :json_files: json files
        """
        json_files = []
        for file in os.listdir(path_file):
            if file.endswith(".json"):
                json_files.append(file)
        return json_files, path_file

    def list_images_from_json(self, json_files, path_file):
        """
        List all images from json files
        list_images_from_json(json_files) -> images
        parameter:
            :param json_files: json files
        return:
            :images: images
        """
        images = []
        for file in json_files:
            json_data = self.load_json("%s/%s" % (path_file, file))
            image = self.image_from_json(json_data)
            images.append(image)
        return images



if __name__ == '__main__':
    load_image = LoadImage()