"""
Generate the visualization results by loading the saved models. The visualization figures are used for report.
"""
import os

import cv2
import torch
import numpy as np

import Utils
from Saved_generator_model import load_saved_generator


def restore_image(model_path, upscale_factor, input_LR_image_path, SR_image_name):
    """
    Load the trained SRResnet (or the generator) model and input a manually selected LR image.
    The model will save the resulting SR image in the images folder.
    :param model_path: The file path of the saved model.
    :param upscale_factor: upscale factor from 2 to 4.
    :param input_LR_image_path: File path of the input LR image. Cropped size is 90, 60, and 45 corresponding to X2, X3, X4.
    :param SR_image_name: The name of the reconstructed image.
    """
    # Load the model
    model_G = load_saved_generator(model_path, upscale_factor=upscale_factor)

    SR_width = 180
    SR_height = 180
    device = torch.device("cpu")
    Tensor = torch.Tensor

    # Get the image input
    # Select the testing image 0886
    # LR_img_path = "Datasets/test/LR_bicubic_X2/0886x2.png"
    input_LR_img = cv2.imread(input_LR_image_path).astype(np.float32) / 255

    h, w, c = input_LR_img.shape
    print('width: ', w)
    print('height: ', h)
    print('channel: ', c)

    # 300 and 600 are the coordinates to crop the HR image. The coordinates shrink according to the upscale factors.
    # input_LR_crop = Utils.img_custom_crop(input_LR_img, SR_height // upscale_factor, SR_width // upscale_factor,
    #                                       300//upscale_factor, 600//upscale_factor)

    input_LR_crop = Utils.img_center_crop(input_LR_img, SR_height // upscale_factor, SR_width // upscale_factor,)

    input_LR_crop = cv2.cvtColor(input_LR_crop, cv2.COLOR_BGR2RGB)
    input_LR_crop = input_LR_crop.transpose(2, 0, 1)

    input_LR_crop = torch.Tensor(input_LR_crop)
    input_LR_crop = input_LR_crop.unsqueeze(0)

    SR_img = model_G(input_LR_crop)

    SR_img = SR_img.detach().numpy()
    SR_img = SR_img.astype(np.float32) * 255
    SR_img = SR_img.squeeze(0)
    SR_img = SR_img.transpose(1, 2, 0)

    SR_sample_folder = os.path.join("images", "ReconstructedSamples")
    if not os.path.exists(SR_sample_folder):
        os.makedirs(SR_sample_folder)
    save_path = os.path.join(SR_sample_folder, SR_image_name)
    cv2.imwrite(save_path, cv2.cvtColor(SR_img, cv2.COLOR_RGB2BGR))

    return


def line_the_presentation_image():
    HR_img_path = "Datasets/test/HR/0886.png"
    HR_img = cv2.imread(HR_img_path)
    line_thickness = 4
    cv2.line(HR_img, (600, 300), (780, 300), (0,0,255), thickness=line_thickness)
    cv2.line(HR_img, (600, 300), (600, 480), (0,0,255), thickness=line_thickness)
    cv2.line(HR_img, (780, 300), (780, 480), (0,0,255), thickness=line_thickness)
    cv2.line(HR_img, (600, 480), (780, 480), (0,0,255), thickness=line_thickness)

    cv2.imwrite("lined_HR_img.png", HR_img)

if __name__ == '__main__':
    # line_the_presentation_image()

    # Use saved model to reconstruct the LR input images and save the results.
    path = "Datasets/test/LR_unknown_X4/0886x4.png"
    # LR_img_path_list = ["Datasets/test/LR_bicubic_X2/0886x2.png", "Datasets/test/LR_bicubic_X3/0886x3.png",
    #                     "Datasets/test/LR_bicubic_X4/0886x4.png", "Datasets/test/LR_unknown_X2/0886x2.png",
    #                     "Datasets/test/LR_unknown_X3/0886x3.png", "Datasets/test/LR_unknown_X4/0886x4.png"]

    LR_img_path_list = ["Datasets/test/LR_bicubic_X2/0890x2.png", "Datasets/test/LR_bicubic_X3/0890x3.png",
                        "Datasets/test/LR_bicubic_X4/0890x4.png", "Datasets/test/LR_unknown_X2/0890x2.png",
                        "Datasets/test/LR_unknown_X3/0890x3.png", "Datasets/test/LR_unknown_X4/0890x4.png"]

    SR_name_list = ["SRResnet_0886_BicubicX2.png", "SRResnet_0886_BicubicX3.png", "SRResnet_0886_BicubicX4.png",
                    "SRResnet_0886_UnknownX2.png", "SRResnet_0886_UnknownX3.png", "SRResnet_0886_UnknownX4.png",
                    "SRGAN_0886_BicubicX2.png", "SRGAN_0886_BicubicX3.png", "SRGAN_0886_BicubicX4.png",
                    "SRGAN_0886_UnknownX2.png", "SRGAN_0886_UnknownX3.png", "SRGAN_0886_UnknownX4.png"]

    restore_image("results/SRGAN/BicubicX3/gen_bestPSNR_seed10309889058753148033.pth.tar", 3,
                  LR_img_path_list[1], SR_name_list[7])

    restore_image("results/SRResnet/BicubicX3/gen_bestPSNR_seed8443402548718658060.pth.tar", 3,
                  LR_img_path_list[1], SR_name_list[1])

    restore_image("results/SRGAN/BicubicX4/gen_bestPSNR_seed1450746893368446039.pth.tar", 4,
                  LR_img_path_list[2], SR_name_list[8])

    restore_image("results/SRResnet/BicubicX4/gen_bestPSNR_seed9753949578282950113.pth.tar", 4,
                  LR_img_path_list[2], SR_name_list[2])

    restore_image("results/SRGAN/UnknownX3/gen_bestPSNR_seed18100551163404145621.pth.tar", 3,
                  LR_img_path_list[4], SR_name_list[10])

    restore_image("results/SRResnet/UnknownX3/gen_bestPSNR_seed3729227894521197293.pth.tar", 3,
                  LR_img_path_list[4], SR_name_list[4])

    restore_image("results/SRGAN/UnknownX2/gen_bestPSNR_seed14691525234421985574.pth.tar", 2,
                  LR_img_path_list[3], SR_name_list[9])

    restore_image("results/SRResnet/UnknownX2/gen_bestPSNR_seed15502664131349655916.pth.tar", 2,
                  LR_img_path_list[3], SR_name_list[3])
