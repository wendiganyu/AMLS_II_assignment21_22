"""
Some helper functions.
"""

import random
import os
import numpy as np
import cv2
import torch
from PIL import Image

import Model_GAN


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

def img_custom_crop(image, target_height, target_width, top_coord, left_coord):
    """
    Crop an input image with custom. For the visualization to check the results after model is trained.
    :param image: input image as an array.
    :param target_height: height of cropped image.
    :param target_width: width of cropped image.
    :param top_coord: Top coordinate of the cropped image.
    :param left_coord: Left cooredinate of the cropped image.
    :return: Cropped image as an array.
    """

    img_crop = image[top_coord:top_coord + target_height, left_coord: left_coord + target_width]

    return img_crop

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


def check_img_size(folder_path):
    """
    Read the size of all image files in a folder, return the minimal height and the minimal width, and the max width.
    :param folder_path: The path of the target folder.
    :return: the minimal height and the minimal width, and the max width.
    """
    img_file_paths = [os.path.join(folder_path, img_file_name) for img_file_name in os.listdir(folder_path)]

    min_height = 100000
    min_width = 100000
    max_width = 0

    for path in img_file_paths:
        img = cv2.imread(path)

        h,w,c = img.shape
        # print('width: ', w)
        # print('height: ', h)
        # print('channel: ', c)

        if h < min_height:
            min_height = h
        if w < min_width:
            min_width = w
        if w > max_width:
            max_width = w

    print("min height: ", min_height)
    print("min width: ", min_width)
    print("max width: ", max_width)

def img_rgb2ycbcr(image, sep_y_channel):
    """
    Simulate rgb2ycbcr function in Matlab to transform the image from RGB space to YCbCr space.
    :param image: input RGB image.
    :param sep_y_channel: It True, extract Y channel separately.
    :return: Image as YCbCr format.
    """

    if sep_y_channel:
        output = np.dot(image, [65.481, 128.553, 24.966]) + 16.0
    else:
        output = np.matmal(image, [[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]) + [16, 128, 128]

    output /= 255.
    output = output.astype(np.float32)

    return output

def load_saved_generator(PATH, upscale_factor):
    """
    Load the saved trained generator in the GAN or SRResnet (both are with the same structure).
    Do the experiment to input a W*H*3 RGB low-resolution image, and output the reconstructed 180*180*3 SR image
    with the saved generator model.
    :param PATH: The path that stores the saved generator model.
    :param upscale_factor: The upscale factor between the LR input and the SR output. Choices are 2,3,4.
    """
    model_G = Model_GAN.Generator(upscale_factor=upscale_factor)
    check_point = torch.load(PATH, map_location=torch.device("cpu"))
    model_G.load_state_dict(check_point["state_dict"])

    # print("epoch:" , check_point["epoch"])
    # # Turns model to evaluate mode.
    # model_G.eval()
    return model_G


if __name__ == '__main__':
    # rotate_dataset()
    # train_HR: min_height:648, min_width:2040
    # valid_HR: min_height:816, min_width:2040
    # test_HR: min_height:1068, min_width:2040
    # check_img_size("Datasets/train/HR")
    # check_img_size("Datasets/valid/HR")
    # check_img_size("Datasets/test/HR")

    HR_img = cv2.imread("Datasets/test/HR/0890.png")
    HR_img_cropped = img_center_crop(HR_img, 180, 180)
    cv2.imwrite("images/HR_0890_cropped.png", HR_img_cropped)
