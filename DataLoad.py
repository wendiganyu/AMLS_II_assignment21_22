"""
This file implements the functions to load DIV2K 2017 datasets.
DIV2K can be downloaded from https://data.vision.ee.ethz.ch/cvl/DIV2K.
"""
import os
import cv2
import numpy as np

from torch.utils.data import Dataset, DataLoader
class DIV2KDatasets(Dataset):
    """
    A class for loading DIV2K image data in PyTorch.
    """
    def __init__(self, LR_folder_path, HR_folder_path, img_crop_height, img_crop_width, upscale_factor, random_crop_trigger):
        """
        Variation values definition.
        :param LR_folder_path: Dataset folder path of low resolution images.
        :param HR_folder_path: Dataset folder path of high resolution images.
        :param img_crop_height: Images in the dataset folder do not have a united size, so set a common size for crop.
        :param img_crop_width: Images in the dataset folder do not have a united size, so set a common size for crop.
        :param upscale_factor: Image up scale factor.
        :param random_crop_trigger: True for train data. Randomly crop the train data for data enhancement.
        """
        super(DIV2KDatasets,self).__init__()

        # Find the folder's file names
        self.LR_file_names = [os.path.join(LR_folder_path, img_file_name) for img_file_name in os.listdir(LR_folder_path)]
        self.HR_file_names = [os.path.join(HR_folder_path, img_file_name) for img_file_name in os.listdir(HR_folder_path)]

        self.img_crop_height = img_crop_height
        self.img_crop_width = img_crop_width
        self.upscale_factor = upscale_factor
        self.random_crop_trigger = random_crop_trigger

    def __getitem__(self, index):
        """
        Read a batch of image data.
        :param index: Index of the image file in the file list.
        :return: The pair of LR image and HR image as a dictionary.
        """
        # Read the image pixels and transform to range [0,1]
        HR_image = cv2.imread(self.HR_file_names[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255
        LR_image = cv2.imread(self.LR_file_names[index], cv2.IMREAD_UNCHANGED).astype(np.float32) / 255

        # HR_image_crop =

if __name__ == '__main__':
    array = np.ones(shape=(2,3,4))
    print(array[0:2, 0:2, ...])
