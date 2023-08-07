import matplotlib.pyplot as plt
import numpy as np

def draw_x_and_plus_on_image(image_path, x_coordinates, y_coordinates):
    """
    Draw an 'X' and '+' on the input image using Matplotlib.

    :param image_path: The path to the input image.
    :param x_coordinates: List of x-coordinates for the 'X' marks.
    :param y_coordinates: List of y-coordinates for the 'X' marks.
    """
    # Load the image using Matplotlib
    image = plt.imread(image_path)

    # Plot the image
    plt.imshow(image)
    plt.axis('off')

    # Draw 'X' marks on the image
    for x, y in zip(x_coordinates, y_coordinates):
        # Draw the 'X' mark
        plt.plot([x - 10, x + 10], [y - 10, y + 10], color='red', linewidth=2)
        plt.plot([x - 10, x + 10], [y + 10, y - 10], color='red', linewidth=2)

        # Draw the '+' mark
        plt.plot([x - 10, x + 10], [y, y], color='green', linewidth=2)
        plt.plot([x, x], [y - 10, y + 10], color='green', linewidth=2)

    # Show the image with the marks
    plt.show()

# Example usage:
input_image_path = "C:/Users/Adiel/Desktop/mobileye-aab_dev/my_test/aachen_000031_000019_leftImg8bit.png"

# Example coordinates for 'X' and '+' marks (you can replace these with your desired coordinates)
x_coordinates = [100, 200, 300]
y_coordinates = [200, 100, 300]

# Draw 'X' and '+' on the image
draw_x_and_plus_on_image(input_image_path, x_coordinates, y_coordinates)


