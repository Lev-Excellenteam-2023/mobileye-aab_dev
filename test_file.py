from PIL import Image, ImageFilter
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import maximum_filter
from sklearn.cluster import DBSCAN

image_path = '/Users/alex/Documents/mobileye-aab_dev/images_set/aachen_000002_000019_leftimg8bit.png'
# Read the input image
input_image = Image.open(image_path)

# Convert the input image to a numpy array
c_image = np.array(input_image, dtype=np.float32) / 255.0
plt.imshow(c_image, cmap='gray')


def IdentifyRedObjects(input_image):
    width, height, rgb = input_image.shape

    # Loop through each pixel
    for y in range(height - 1):
        for x in range(width - 1):
            # Get the color value(s) at the current pixel
            # For RGB images, you'll get (R, G, B) values
            # For grayscale images, you'll get a single intensity value
            pixel = input_image[x][y]
            r, g, b = pixel
            if r > g and r > b:
                input_image[x][y][0] = 255
            #else:
                #input_image[x][y] = 0, 0, 0

    return input_image

def draw_x_and_plus_on_image(image_path, cordinates):
    """
    Draw an 'X' and '+' on the input image using Matplotlib.

    :param image_path: The path to the input image.
    :param coordinates: List of lists for red-coordinates for the 'X' marks and green-coordinates for '+' marks.
    """
    # Load the image using Matplotlib
    image = plt.imread(image_path)

    # Plot the image
    plt.imshow(image)
    plt.axis('off')

    # Draw 'X' marks on the image
    for y, x,r in cordinates[0]:
        # Draw the 'X' mark
        plt.plot([x - 10, x + 10], [y - 10, y + 10], color='red', linewidth=2)
        plt.plot([x - 10, x + 10], [y + 10, y - 10], color='red', linewidth=2)

    for y, x,r in cordinates[1]:
        # Draw the '+' mark
        plt.plot([x - 10, x + 10], [y, y], color='green', linewidth=2)
        plt.plot([x, x], [y - 10, y + 10], color='green', linewidth=2)

    # Show the image with the marks
    plt.show()


def calculate_radius_of_red_pixels(image, points):
    row, col, color = image.shape
    new_red_points = []
    max_radius = 35
    for point in points:
        right_sum = 0
        left_sum = 0
        y, x = point
        for index in range(1, max_radius + 1):
            if x + index <= col:
                r1, g1, b1 = image[y][x + index]
                if r1 >= g1 and r1 >= b1:
                    right_sum += 1
                else:
                    break
            else:
                break
        for index in range(1, max_radius + 1):
            if x - index >= 0:
                r1, g1, b1 = image[y][x - index]
                if r1 >= g1 and r1 >= b1:
                    left_sum += 1
                else:
                    break
            else:
                break
        total_sum = right_sum + left_sum
        if right_sum + left_sum < max_radius:
            new_red_points.append([y, x, total_sum + 5])
    return new_red_points


def calculate_radius_of_green_pixels(image, points):
    row, col, color = image.shape
    new_green_points = []
    max_radius = 35
    for point in points:
        right_sum = 0
        left_sum = 0
        y, x = point
        for index in range(1, max_radius + 1):
            if x + index <= col:
                r1, g1, b1 = image[y][x + index]
                if g1 > r1:
                    right_sum += 1
                else:
                    break
            else:
                break
        for index in range(1, max_radius + 1):
            if x - index >= 0:
                r1, g1, b1 = image[y][x - index]
                if r1 >= g1 and r1 >= b1:
                    left_sum += 1
                else:
                    break
            else:
                break
        total_sum = right_sum + left_sum
        if right_sum + left_sum < max_radius:
            new_green_points.append([y, x, total_sum + 5])
    return new_green_points



#output_image = IdentifyRedObjects(c_image)

#plt.imshow(output_image, cmap='gray')


sigma = 1.5  # Adjust the sigma value based on the size of the circles you want to detect


circle_kernel = np.array([[-2, -2, -2, -2, -2],
                          [-2, -2, 10, -2, -2],
                          [-2, 10, 10, 10, -2],
                          [-2, -2, 10, -2, -2],
                          [-2, -2, -2, -2, -2]])



# Normalize the kernel
circle_kernel = circle_kernel / (2 * np.pi * sigma ** 2)

# Apply convolution to the red channel using convolve2d
filtered = convolve2d(c_image[:, :, 0], circle_kernel, mode='same', boundary='wrap')

plt.imshow(filtered, cmap='gray')


local_maxima = maximum_filter(filtered, footprint=np.ones((2, 2)))
plt.imshow(local_maxima, cmap='gray')
# Find the maximum color value among the local maxima
max_value = np.max(local_maxima)

# Define a threshold distance for "near" pixels
threshold_distance = 0.65  # Adjust this based on your needs
# Find the positions of pixels near the maximum color value
near_positions = np.argwhere((local_maxima >= max_value - threshold_distance) & (local_maxima <= max_value))

# Apply DBSCAN clustering
dbscan = DBSCAN(eps=5, min_samples=1)
clusters = dbscan.fit_predict(near_positions)

# Initialize a dictionary to store cluster centroids
cluster_centroids = {}

# Calculate cluster centroids
for cluster_label in np.unique(clusters):
    cluster_indices = np.where(clusters == cluster_label)[0]
    cluster_points = near_positions[cluster_indices]
    cluster_centroid = np.mean(cluster_points, axis=0).astype(int)
    cluster_centroids[cluster_label] = cluster_centroid

# Print the positions and color values of representative pixels (cluster centroids)
for cluster_label, centroid in cluster_centroids.items():
    row, col = centroid
    print(f"Cluster {cluster_label}:")
    print(f"Position: Row {row}, Column {col}")
    print(f"Color Value: {c_image[row, col]}")


# Initialize a dictionary to store cluster centroids
ended_red_pixels = []
ended_green_pixels = []
ended_pixels = [ended_red_pixels, ended_green_pixels]
# Print the positions and color values of representative pixels (cluster centroids)
for cluster_label, centroid in cluster_centroids.items():
    row, col = centroid
    pixel = c_image[row][col]
    r, g, b = pixel
    if r >= g and r > b:
        ended_pixels[0].append(centroid)
    if g > r:
        ended_pixels[1].append(centroid)

new_ended_points = []
new_red_points = calculate_radius_of_red_pixels(c_image, ended_pixels[0])
new_green_points = calculate_radius_of_green_pixels(c_image, ended_pixels[1])
new_ended_points.append(new_red_points)
new_ended_points.append(new_green_points)

draw_x_and_plus_on_image(image_path, new_ended_points)

print("Goog bay!")




