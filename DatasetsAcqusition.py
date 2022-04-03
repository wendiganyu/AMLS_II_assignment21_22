"""
This file implements the functions to load DIV2K 2017 datasets.
DIV2K can be downloaded from https://data.vision.ee.ethz.ch/cvl/DIV2K.
"""
import os

from torch.utils.data import Dataset, DataLoader
class DIV2KDatasets(Dataset):
    """
    A class for loading DIV2K image data in PyTorch.
    """
    def __init__(self, LR_folder_path, HR_folder_path, img_crop_size, upscale_factor, random_crop_trigger):
        """
        Variation values definition.
        :param LR_folder_path: Dataset folder path of low resolution images.
        :param HR_folder_path: Dataset folder path of high resolution images.
        :param img_crop_size: Images in the dataset folder do not have a united size, so set a size for crop.
        :param upscale_factor: Image up scale factor.
        :param random_crop_trigger: True for train data. Randomly crop the train data for data enhancement.
        """
        super(DIV2KDatasets,self).__init__()

        # Find the folder's file names
        self.LR_file_names = [os.path.join(LR_folder_path, img_file_name) for img_file_name in os.listdir(LR_folder_path)]
        self.HR_file_names = [os.path.join(HR_folder_path, img_file_name) for img_file_name in os.listdir(HR_folder_path)]

        self.img_crop_size = img_crop_size # width == length
        self.upscale_factor = upscale_factor
        self.random_crop_trigger = random_crop_trigger

