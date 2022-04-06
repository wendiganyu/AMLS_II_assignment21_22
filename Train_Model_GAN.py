"""
Run this file to train the GAN model for image super resolution.
"""
import os
import shutil
import time
from enum import Enum

import DataLoad

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.cuda import amp
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from tqdm.auto import tqdm
import Model_GAN


def train_GAN_model():
    # -------------------------------------------------------------------------------------------------
    # Initial settings

    # Network evaluation indicator
    # Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between the maximum possible power of a
    # signal and the power of corrupting noise that affects the fidelity of its representation. Because many signals
    # have a very wide dynamic range, PSNR is usually expressed as a logarithmic quantity using the decibel scale.
    best_psnr = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 100
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    # -------------------------------------------------------------------------------------------------
    # Create PyTorch Dataloaders
    img_crop_height = 720
    img_crop_width = 720
    train_datasets = DataLoad.DIV2KDatasets("Datasets/train/LR_bicubic_X2", "Datasets/train/HR", img_crop_height, img_crop_width, upscale_factor=2, random_crop_trigger=True)
    valid_datasets = DataLoad.DIV2KDatasets("Datasets/valid/LR_bicubic_X2", "Datasets/valid/HR", img_crop_height, img_crop_width, upscale_factor=2, random_crop_trigger=False)
    test_datasets = DataLoad.DIV2KDatasets("Datasets/test/LR_bicubic_X2", "Datasets/test/HR", img_crop_height, img_crop_width, upscale_factor=2, random_crop_trigger=False)

    train_loader = DataLoader(train_datasets,
                              batch_size=1,
                              shuffle=True)
    valid_loader = DataLoader(valid_datasets,
                              batch_size=1,
                              shuffle=False)
    test_loader = DataLoader(test_datasets,
                             batch_size=1,
                             shuffle=False)

    # -------------------------------------------------------------------------------------------------
    # Load model
    generator = Model_GAN.Generator().to(device)
    discriminator = Model_GAN.Discriminator().to(device)

    # Total params of the generator
    total_params = sum(p.numel() for p in generator.parameters())
    print("Total parameters of the generator: " + str(total_params))
    # Total trainable params of the generator
    total_trainable_params = sum(p.numel() for p in generator.parameters() if p.requires_grad)
    print("Total trainable parameters of generator: " + str(total_trainable_params))

    # Total params of the discriminator
    total_params = sum(p.numel() for p in discriminator.parameters())
    print("Total parameters of the discriminator: " + str(total_params))
    # Total trainable params of the discriminator
    total_trainable_params = sum(p.numel() for p in discriminator.parameters() if p.requires_grad)
    print("Total trainable parameters of discriminator: " + str(total_trainable_params))

    # -------------------------------------------------------------------------------------------------
    # Define loss criterion.
    psnr_loss_criterion = nn.MSELoss().to(device)
    pixel_loss_criterion = nn.MSELoss().to(device)
    content_loss_criterion = Model_GAN.ContentLoss().to(device)
    adversarial_loss_criterion = nn.BCEWithLogitsLoss().to(device)

    # -------------------------------------------------------------------------------------------------
    # Define optimizer

    learning_rate = 1e-4
    gen_optimizer = optim.Adam(generator.parameters(), learning_rate)
    dis_optimizer = optim.Adam(discriminator.parameters(), learning_rate)

    # -------------------------------------------------------------------------------------------------
    # Define scheduler
    # Multiply LR by gamma=0.1 every epoch_num//2 epochs.
    dis_scheduler = lr_scheduler.StepLR(dis_optimizer, step_size=epoch_num//2, gamma=0.1)
    gen_scheduler = lr_scheduler.StepLR(gen_optimizer, step_size=epoch_num//2, gamma=0.1)

    # -------------------------------------------------------------------------------------------------
    # Create a folder to store some SR result samples.
    model_name = "SRGAN"
    sample_folder = os.path.join("samples", model_name)
    result_folder = os.path.join("results",model_name)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # -------------------------------------------------------------------------------------------------
    # Create summary writers.
    writer = SummaryWriter(os.path.join("sample", "logs",model_name))

    # Train
    for epoch in range(epoch_num):
        generator.train()
        discriminator.train()

        for batch_idx, imgs in enumerate(tqdm(train_loader)):
            # Configure input
            LR_imgs = Variable(imgs["LR"].type(Tensor))
            HR_imgs = Variable(imgs["HR"].type(Tensor))

            # Set the labels
            real_img_label = torch.full([LR_imgs.size(0), 1], 1.0, dtype=LR_imgs.dtype, device=device)
            fake_img_label = torch.full([LR_imgs.size(0), 1], 0.0, dtype=LR_imgs.dtype, device=device)

            # Generator
            SR_imgs = generator(LR_imgs)

            # Train Discriminator
            for para in discriminator.parameters():
                para.requires_grad = True

            dis_optimizer.zero_grad()

            # Calculate the loss of discriminator on HR images
            HR_output = discriminator(HR_imgs)
            dis_loss_HR = adversarial_loss_criterion(HR_output, real_img_label)

            dis_loss_HR.backward()

            # Calculate the loss of discriminator on SR images
            SR_output = discriminator(SR_imgs.detach())
            dis_loss_SR = adversarial_loss_criterion(SR_output, fake_img_label)

            dis_loss_SR.backward()

            # Update discriminator parameters
            dis_optimizer.step()

            # Add two losses of discriminator
            dis_loss_total = dis_loss_HR + dis_loss_SR

            #-------------------------------------------
            # Train generator
            # Close the updating of discriminator first
            for para in discriminator.parameters():
                para.requires_grad = False

            gen_optimizer.zero_grad()

            SR_dis_valid_output = discriminator(SR_imgs)

            # Define weights of different losses
            pixel_loss_weight = 1.0
            content_loss_weight = 1.0
            adversarial_loss_weight = 0.001

            # Calculate losses
            pixel_loss = pixel_loss_weight * pixel_loss_criterion(SR_imgs, HR_imgs.detach())
            content_loss = content_loss_weight * content_loss_criterion(SR_imgs, HR_imgs.detach())
            adversarial_loss = adversarial_loss_weight * adversarial_loss_criterion(SR_dis_valid_output, real_img_label)

            gen_loss_total = pixel_loss + content_loss + adversarial_loss

            gen_loss_total.backward()

            gen_optimizer.step()


            # Calculate scores of HR and SR images on discriminator
            dis_HR_prob = torch.sigmoid(torch.mean(HR_output))
            dis_SR_prob = torch.sigmoid(torch.mean(SR_output))

            # Accuracy
            psnr = 10.0 * torch.log10(1.0 / psnr_loss_criterion(SR_imgs, HR_imgs))
            print("PSNR:", psnr)

        #------------------------------------------------------------------------------------------------
        # Validate











    return

#
# def validate(model, valid_loader, psnr_criterion, epoch, writer, device, mode):
#
#     # Put the model in verification mode
#     model.eval()
#
#     with torch.no_grad():
#
#         for batch_idx, imgs in enumerate(tqdm(valid_loader)):
#             # Configure input
#             LR_imgs = imgs["LR"].to(device)
#             HR_imgs = imgs["HR"].to(device)
#
#             SR_imgs = model(LR_imgs)
#
#             # Convert RGB tensor to Y tensor
#             sr_image = imgproc.tensor2image(sr, range_norm=False, half=True)
#             sr_image = sr_image.astype(np.float32) / 255.
#             sr_y_image = imgproc.rgb2ycbcr(sr_image, use_y_channel=True)
#             sr_y_tensor = imgproc.image2tensor(sr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
#
#             hr_image = imgproc.tensor2image(hr, range_norm=False, half=True)
#             hr_image = hr_image.astype(np.float32) / 255.
#             hr_y_image = imgproc.rgb2ycbcr(hr_image, use_y_channel=True)
#             hr_y_tensor = imgproc.image2tensor(hr_y_image, range_norm=False, half=True).to(config.device).unsqueeze_(0)
#
#             # measure accuracy and record loss
#             psnr = 10. * torch.log10(1. / psnr_criterion(sr_y_tensor, hr_y_tensor))
#             psnres.update(psnr.item(), lr.size(0))
#
#
#             # Record training log information
#             if batch_index % config.print_frequency == 0:
#                 progress.display(batch_index)
#
#             # Preload the next batch of data
#             batch_data = valid_prefetcher.next()
#
#             # After a batch of data is calculated, add 1 to the number of batches
#             batch_index += 1
#
#
#     if mode == "Valid":
#         writer.add_scalar("Valid/PSNR", psnres.avg, epoch + 1)
#     elif mode == "Test":
#         writer.add_scalar("Test/PSNR", psnres.avg, epoch + 1)
#     else:
#         raise ValueError("Unsupported mode, please use `Valid` or `Test`.")
#
#     return psnres.avg

if __name__ == '__main__':
    train_GAN_model()