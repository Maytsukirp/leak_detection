from modules.load_image import LoadImage


def test_load_image():
    # Create object LoadImage
    load_image = LoadImage()
    json_data = load_image.load_json('resources/json/0_MOV_0881blue marlin.json')
    image = load_image.image_from_json(json_data)
    assert image.shape == (240, 320, 3)
    print('Test load_image: OK')
    load_image.show_image(image)




