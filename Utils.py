"""
Some helper functions.
"""

import random

def img_center_crop(image, target_height, target_width):
    """
    Crop an input image from center
    :param image: input image as an array.
    :param target_height: height of cropped image.
    :param target_width: width of cropped image.
    :return: Cropped image as an array.
    """
    org_img_height, org_img_width = image.shape[:2]

    top_coord = (org_img_height - target_height) // 2
    left_coord = (org_img_width - target_width) // 2

    img_crop = image[top_coord:top_coord + target_height, left_coord: left_coord + target_width]

    return img_crop


def img_random_crop(image, target_height, target_width, top_coord = None, left_coord = None):
    """
    Crop an input image randomly. For data enhancement in data training.
    :param image: input image as an array.
    :param target_height: height of cropped image.
    :param target_width: width of cropped image.
    :return: Cropped image as an array.
    """
    org_img_height, org_img_width = image.shape[:2]

    # Randomly set the top and left coordinates.
    if ((top_coord is None) and (left_coord is None)):
        top_coord = random.randint(0, org_img_height - target_height)
        left_coord = random.randint(0, org_img_width - target_width)

    img_crop = image[top_coord:top_coord + target_height, left_coord: left_coord + target_width]

    return img_crop, top_coord, left_coord