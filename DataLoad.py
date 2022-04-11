"""
This file implements the functions to load DIV2K 2017 datasets.
DIV2K can be downloaded from https://data.vision.ee.ethz.ch/cvl/DIV2K.
"""
import os
import cv2
import numpy as np
from torchvision.transforms import functional as F

import Utils

from torch.utils.data import Dataset, DataLoader
class DIV2KDatasets(Dataset):
    """
    A class for loading DIV2K image data in PyTorch.
    """
    def __init__(self, LR_folder_path, HR_folder_path, HR_img_crop_height, HR_img_crop_width, upscale_factor, random_crop_trigger):
        """
        Variation values definition.
        :param LR_folder_path: Dataset folder path of low resolution images.
        :param HR_folder_path: Dataset folder path of high resolution images.
        :param HR_img_crop_height: Images in the dataset folder do not have a united size, so set a common size for crop.
        :param HR_img_crop_width: Images in the dataset folder do not have a united size, so set a common size for crop.
        :param upscale_factor: Image up scale factor.
        :param random_crop_trigger: True for train data. Randomly crop the train data for data enhancement.
        """
        super(DIV2KDatasets,self).__init__()

        # Find the folder's file names
        self.LR_file_names = [os.path.join(LR_folder_path, img_file_name) for img_file_name in os.listdir(LR_folder_path)]
        self.HR_file_names = [os.path.join(HR_folder_path, img_file_name) for img_file_name in os.listdir(HR_folder_path)]

        self.HR_img_crop_height = HR_img_crop_height
        self.HR_img_crop_width = HR_img_crop_width
        self.upscale_factor = upscale_factor
        self.random_crop_trigger = random_crop_trigger

    def __getitem__(self, index):
        """
        Read a batch of image data.
        :param index: Index of the image file in the file list.
        :return: The pair of LR image and HR image as a dictionary.
        """
        # Read the image pixels and transform to range [0,1]
        HR_image = cv2.imread(self.HR_file_names[index]).astype(np.float32) / 255
        LR_image = cv2.imread(self.LR_file_names[index]).astype(np.float32) / 255
        print(f"HR image shape of {self.HR_file_names[index]}:", HR_image.shape)

        LR_img_crop_height = self.HR_img_crop_height // self.upscale_factor
        LR_img_crop_width = self.HR_img_crop_width // self.upscale_factor

        if self.random_crop_trigger:
            HR_image_crop, HR_top_coord, HR_left_coord = Utils.img_random_crop(HR_image, self.HR_img_crop_height, self.HR_img_crop_width)
            LR_image_crop, _, _ = Utils.img_random_crop(LR_image, LR_img_crop_height, LR_img_crop_width,
                                                        HR_top_coord // self.upscale_factor, HR_left_coord // self.upscale_factor)
        else:
            HR_image_crop = Utils.img_center_crop(HR_image, self.HR_img_crop_height, self.HR_img_crop_width)
            LR_image_crop = Utils.img_center_crop(LR_image, LR_img_crop_height, LR_img_crop_width)

        HR_image_crop = cv2.cvtColor(HR_image_crop, cv2.COLOR_BGR2RGB)
        LR_image_crop = cv2.cvtColor(LR_image_crop, cv2.COLOR_BGR2RGB)

        return {"HR": F.to_tensor(HR_image_crop),
                "LR": F.to_tensor(LR_image_crop)
                }

    def __len__(self):
        return len(self.HR_file_names)


if __name__ == '__main__':
    array = np.ones(shape=(2,3,4))
    print(array[0:2, 0:2, ...])
