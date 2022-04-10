"""
Some helper functions.
"""

import random
import os
import numpy as np
import cv2
from PIL import Image

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

def rotate_target_img(folder_path):
    """
    Retrieve the dataset files and rotate an image by 90 degree if its height > width.
    :param folder_path: target folder's path.
    """
    img_file_paths = [os.path.join(folder_path, img_file_name) for img_file_name in os.listdir(folder_path)]

    for path in img_file_paths:
        img = cv2.imread(path)

        h,w,c = img.shape
        # print('width: ', w)
        # print('height: ', h)
        # print('channel: ', c)

        # img = Image.open(path)
        #
        # w = img.width
        # h = img.height
        # # print("width: ", w)
        # # print("height: ", h)

        if(h > w):
            print(f"Height{h} > width{w}: ", path )
            img_rotate = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(path, img_rotate)

def rotate_dataset():
    """
    Use rotate_target_image function to do the functionality to the whole dataset.
    """
    dataset_folders = ["train", "valid", "test"]
    sub_folders = ["HR", "LR_bicubic_X2", "LR_bicubic_X3", "LR_bicubic_X4", "LR_unknown_X2", "LR_unknown_X3", "LR_unknown_X4"]

    for folder in dataset_folders:
        for sub_folder in sub_folders:
            path = os.path.join("Datasets", folder, sub_folder)
            rotate_target_img(path)


def check_img_size_min(folder_path):
    """
    Read the size of all image files in a folder, return the minimal height and the minimal width.
    :param folder_path: The path of the target folder.
    :return: the minimal height and the minimal width.
    """
    img_file_paths = [os.path.join(folder_path, img_file_name) for img_file_name in os.listdir(folder_path)]

    min_height = 100000
    min_width = 100000

    for path in img_file_paths:
        img = cv2.imread(path)

        h,w,c = img.shape
        # print('width: ', w)
        # print('height: ', h)
        # print('channel: ', c)

        if h < min_width:
            min_height = h
        if w < min_width:
            min_width = w

    print("min height: ", min_height)
    print("min width: ", min_width)

if __name__ == '__main__':
    rotate_dataset()
    # train_HR: min_height:1332, min_width:2040
    # valid_HR: min_height:1356, min_width:2040
    # test_HR: min_height:1152, min_width:2040
    check_img_size_min("Datasets/train/HR")
    check_img_size_min("Datasets/valid/HR")
    check_img_size_min("Datasets/test/HR")


