"""
Load the saved trained generator in the GAN or SRResnet (both are with the same structure),
and do some visualization experiments to output and save the SR image.
"""

import torch
import Model_GAN
import cv2
import numpy as np
from torch.autograd import Variable
import Utils


def load_saved_generator(PATH, upscale_factor):
    """
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

