'''
OpenCV Basics - Greyscale Exercises
'''
import cv2
import numpy as np

def pixel_count(GS_image):
    """ Calculates the total number of pixels in an image.
    Args:
        GS_image (numpy.ndarray): A grayscale image.
    Returns:
        int: The number of pixels in an image.
    """
    return GS_image.size

def average_pixel_value(GS_image):
    """ Calculates the average pixel value of a grayscale image.
    Args:
        GS_image (numpy.ndarray): A grayscale image.
    Returns:
        int: Average pixel value (truncated) in the image (Range of 0-255).
    """
    total_pixels = GS_image.size
    total_sum = GS_image.sum()
    return int(total_sum // total_pixels)

def to_BW(GS_image):
    """ Converts a grayscale image to a black and white image.
    Args:
        GS_image (numpy.ndarray): A grayscale image.
    Returns:
        numpy.ndarray: The black and white image.
    """
    BW_image = GS_image.copy()
    BW_image[BW_image > 128] = 255
    BW_image[BW_image <= 128] = 0
    return BW_image

def image_average_grayscale(GS_image1, GS_image2):
    """ Averages the pixels of the two grayscale input images.
    Args:
        GS_image1 (numpy.ndarray): A grayscale image.
        GS_image2 (numpy.ndarray): A grayscale image.
    Returns:
        numpy.ndarray: An image which is the average of image1 and image2.
    """
    return ((GS_image1.astype('int16') + GS_image2.astype('int16')) // 2).astype('uint8')

def flip_horizontal_grayscale(GS_image):
    """ Flips the input image across the horizontal axis.
    Args:
        GS_image (numpy.ndarray): A grayscale image.
    Returns:
        numpy.ndarray: The horizontally flipped image.
    """
    return GS_image[:, ::-1]

def histogram_grayscale(GS_image):
    """ Counts the number of pixels of each value (0 -> 255) in the grayscale image.
    Args:
        GS_image (numpy.ndarray): A grayscale image.
    Returns:
        list: A list with 256 elements, representing the pixel count of each grayscale level.
    """
    hist, _ = np.histogram(GS_image, bins=256, range=(0, 256))
    return hist.tolist()

if __name__ == '__main__':
    image1_location = 'roost.png' 
    image2_location = 'roost.png'
    img = cv2.imread(image1_location, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_location, cv2.IMREAD_GRAYSCALE)

    print(f'Pixel Count: {pixel_count(img)}')
    print(f'Average Pixel Value: {average_pixel_value(img)}')
    cv2.imshow(f'{image1_location} - to_BW', to_BW(img))
    cv2.imshow(f'{image1_location} v. {image2_location} - average_grayscale', image_average_grayscale(img, img2))
    cv2.imshow(f'{image1_location} - flip_horizontal_grayscale', flip_horizontal_grayscale(img))
    print(f'Grayscale Histogram: {histogram_grayscale(img)}')

    cv2.waitKey()
    cv2.destroyAllWindows()
