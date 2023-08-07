from PIL import Image


def crop_image(image_path, left, top, right, bottom):
    """
    Crop the input image to the specified region.

    :param image_path: The path to the input image.
    :param left: The x-coordinate of the left edge of the crop box.
    :param top: The y-coordinate of the top edge of the crop box.
    :param right: The x-coordinate of the right edge of the crop box.
    :param bottom: The y-coordinate of the bottom edge of the crop box.
    :return: The cropped image.
    """
    # Open the image using Pillow
    image = Image.open(image_path)

    # Crop the image using the specified coordinates
    cropped_image = image.crop((left, top, right, bottom))

    return cropped_image


# Example usage:
input_image_path = "C:/Users/Adiel/Desktop/mobileye-aab_dev/my_test/aachen_000031_000019_leftImg8bit.png"

# Define the coordinates of the crop box (left, top, right, bottom)
left = 400
top = 100
right = 800
bottom = 500

# Crop the image
cropped_image = crop_image(input_image_path, left, top, right, bottom)

# Save the cropped image to a new file
output_image_path = "path_to_save_cropped_image.jpg"
cropped_image.save(output_image_path)