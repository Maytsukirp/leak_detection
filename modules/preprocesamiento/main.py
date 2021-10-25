

from src.images_processing import ImagesProcessing


def run_test():
    from tests.load_image import test_load_image
    test_load_image()


def run():
    images_processing = ImagesProcessing()
    # cambiar ruta de json a las que se quiere procesar
    images = images_processing.list_images('resources/json/label-images')
    images_processing.images_processing(images)
    images_processing.image_preprocessing_2(images)


if __name__ == "__main__":
    run()