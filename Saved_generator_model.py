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



