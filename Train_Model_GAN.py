"""
Run this file to train the GAN model for image super resolution.
"""
import os

import shutil
import time
from enum import Enum
import argparse
import DataLoad

import numpy as np
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable
from torchvision.transforms import functional as F

from tqdm.auto import tqdm
import Model_GAN
import Utils


def train_GAN_model(LR_train_folder_path, LR_valid_folder_path, LR_test_folder_path, upscale_factor, track_name):
    """
    Train the SRGAN model.
    :param LR_train_folder_path: The dataset folder path of the LR image for train.
    :param LR_valid_folder_path: The dataset folder path of the LR image for valid.
    :param LR_test_folder_path: The dataset folder path of the LR image for test.
    :param upscale_factor: Upscale factor from 2 to 4.
    :param track_name: The track name of the LR images.
    """
    # -------------------------------------------------------------------------------------------------
    # Initial settings

    # Network evaluation indicator
    # Peak signal-to-noise ratio (PSNR) is an engineering term for the ratio between the maximum possible power of a
    # signal and the power of corrupting noise that affects the fidelity of its representation. Because many signals
    # have a very wide dynamic range, PSNR is usually expressed as a logarithmic quantity using the decibel scale.
    best_psnr = 0.0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epoch_num = 300
    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor
    train_batch_size = 20
    log_freq = 10

    # Set a seed to store the model params into a file with unique name.
    seed = torch.initial_seed()
    print(f'Use seed : {seed}')

    # -------------------------------------------------------------------------------------------------
    # Create PyTorch Dataloaders
    img_crop_height = 180
    img_crop_width = 180
    train_datasets = DataLoad.DIV2KDatasets(LR_train_folder_path, "Datasets/train/HR", img_crop_height,
                                            img_crop_width, upscale_factor=upscale_factor, random_crop_trigger=True)
    valid_datasets = DataLoad.DIV2KDatasets(LR_valid_folder_path, "Datasets/valid/HR", img_crop_height,
                                            img_crop_width, upscale_factor=upscale_factor, random_crop_trigger=False)
    test_datasets = DataLoad.DIV2KDatasets(LR_test_folder_path, "Datasets/test/HR", img_crop_height,
                                           img_crop_width, upscale_factor=upscale_factor, random_crop_trigger=False)

    train_loader = DataLoader(train_datasets,
                              batch_size=train_batch_size,
                              shuffle=True)

    train_loader_len = len(train_loader)

    valid_loader = DataLoader(valid_datasets,
                              batch_size=5,
                              shuffle=False)
    test_loader = DataLoader(test_datasets,
                             batch_size=5,
                             shuffle=False)

    # -------------------------------------------------------------------------------------------------
    # Load model
    generator = Model_GAN.Generator(upscale_factor=upscale_factor).to(device)
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
    dis_scheduler = lr_scheduler.StepLR(dis_optimizer, step_size=epoch_num // 5, gamma=0.1)
    gen_scheduler = lr_scheduler.StepLR(gen_optimizer, step_size=epoch_num // 5, gamma=0.1)

    # -------------------------------------------------------------------------------------------------
    # Create a folder to store some SR result samples.
    model_name = "SRGAN"
    sample_folder = os.path.join("summary_writer_records", model_name, track_name, "logs")
    result_folder = os.path.join("results", model_name, track_name)
    if not os.path.exists(sample_folder):
        os.makedirs(sample_folder)
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)

    # -------------------------------------------------------------------------------------------------
    # Create summary writers.
    writer = SummaryWriter(os.path.join("summary_writer_records", model_name, track_name, "logs"))

    # Train
    for epoch in range(epoch_num):
        # -------------------------------------------------------------------------------------
        # Set average meters and progress printer
        avg_meter_pixel_loss = AverageMeter("Pixel loss", ":6.6f")
        avg_meter_content_loss = AverageMeter("Content loss", ":6.6f")
        avg_meter_adversarial_loss = AverageMeter("Adversarial loss", ":6.6f")
        avg_meter_dis_HR_prob = AverageMeter("Probability of Discriminator(HR)", ":6.6f")
        avg_meter_dis_SR_prob = AverageMeter("Probability of Discriminator(SR)", ":6.6f")
        avg_meter_psnr = AverageMeter("PSNR", ":4.2f")
        avg_meter_list = [avg_meter_pixel_loss, avg_meter_content_loss, avg_meter_adversarial_loss,
                          avg_meter_dis_HR_prob,
                          avg_meter_dis_SR_prob, avg_meter_psnr]
        progress = ProgressMeter(train_loader_len, avg_meter_list, prefix=f"Epoch: [{epoch + 1}]")
        # -------------------------------------------------------------------------------------

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

            # -------------------------------------------
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

            # -------------------------------------------------------------------------------------------
            # Calculate scores of HR and SR images on discriminator
            dis_HR_prob = torch.sigmoid(torch.mean(HR_output))
            dis_SR_prob = torch.sigmoid(torch.mean(SR_output))

            # Calculate related metrics
            # print("LR_imgs_Variable_size: ", LR_imgs.size(0))

            psnr = 10.0 * torch.log10(1.0 / psnr_loss_criterion(SR_imgs, HR_imgs))
            avg_meter_pixel_loss.update(pixel_loss.item(), LR_imgs.size(0))
            print("pixel loss: ", pixel_loss.item())
            avg_meter_content_loss.update(content_loss.item(), LR_imgs.size(0))
            avg_meter_adversarial_loss.update(adversarial_loss.item(), LR_imgs.size(0))
            avg_meter_dis_HR_prob.update(dis_HR_prob.item(), LR_imgs.size(0))
            avg_meter_dis_SR_prob.update(dis_SR_prob.item(), LR_imgs.size(0))
            avg_meter_psnr.update(psnr.item(), LR_imgs.size(0))

            # -------------------------------------------------------------------------------------------
            # Record training log information with log frequency
            if batch_idx % log_freq == 0:
                num_iter = batch_idx + epoch * train_loader_len
                writer.add_scalar("Train/Discriminator Loss", dis_loss_total.item(), num_iter)
                writer.add_scalar("Train/Generator Loss", gen_loss_total.item(), num_iter)
                writer.add_scalar("Train/Pixel Loss", pixel_loss.item(), num_iter)
                writer.add_scalar("Train/Content Loss", content_loss.item(), num_iter)
                writer.add_scalar("Train/Adversarial Loss", adversarial_loss.item(), num_iter)
                writer.add_scalar("Train/Probability of D(HR)", dis_HR_prob.item(), num_iter)
                writer.add_scalar("Train/Probability of D(SR)", dis_SR_prob.item(), num_iter)
                writer.add_scalar("Train/PSNR", psnr.item(), num_iter)
                progress.display(batch_idx)
        # ------------------------------------------------------------------------------------------------
        # Validate and test
        PSNR_valid = valid_test(generator, valid_loader, psnr_loss_criterion, epoch, writer, device, "valid", log_freq)
        PSNR_test = valid_test(generator, test_loader, psnr_loss_criterion, epoch, writer, device, "test", log_freq)

        # Update learning rate scheduler
        dis_scheduler.step()
        gen_scheduler.step()

        # ------------------------------------------------------------------------------------------------
        # Save the model with the highest test PSNR and save the model of last epoch
        if PSNR_test > best_psnr:
            best_psnr = max(PSNR_test, best_psnr)

            torch.save({"epoch": epoch + 1,
                        "test_PSNR": PSNR_test,
                        "best_PSNR": best_psnr,
                        "state_dict": generator.state_dict(),
                        "optimizer": gen_optimizer.state_dict(),
                        "scheduler": gen_scheduler.state_dict(),
                        "seed": seed
                        },
                       os.path.join(result_folder, f"gen_bestPSNR_seed{seed}.pth.tar"))

            torch.save({"epoch": epoch + 1,
                        "test_PSNR": PSNR_test,
                        "best_PSNR": best_psnr,
                        "state_dict": discriminator.state_dict(),
                        "optimizer": dis_optimizer.state_dict(),
                        "scheduler": dis_scheduler.state_dict(),
                        "seed": seed
                        },
                       os.path.join(result_folder, f"dis_bestPSNR_seed{seed}.pth.tar"))

    return


def valid_test(model, data_loader, psnr_criterion, epoch, writer, device, mode, log_freq):
    """
    Use the trained model to do the validate or test process.
    :param model: The trained model used.
    :param data_loader: Valid data loader or test data loader.
    :param psnr_criterion: Calculate the PSNR between SR image and HR image.
    :param epoch: Current number of epoch.
    :param writer: Summary writer for writing log information.
    :param device: CPU or GPU.
    :param mode: "valid" or "test".
    :param log_freq: The frequency of writing the log files.
    :return: Average PSNR performance.
    """
    avg_meter_PSNR = AverageMeter("PSNR", ":4.2f")
    progress = ProgressMeter(len(data_loader), [avg_meter_PSNR], prefix=f'{mode}: ')

    # Switch evaluation mode
    model.eval()

    with torch.no_grad():

        for batch_idx, imgs in enumerate(tqdm(data_loader)):
            # Configure input
            LR_imgs = imgs["LR"].to(device)
            HR_imgs = imgs["HR"].to(device)

            SR_imgs = model(LR_imgs)

            # Convert RGB tensor to Y_CB_CR tensor
            # Pytorch tensor to numpy array image
            SR_nparray = SR_imgs.squeeze_(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).cpu().numpy().astype("uint8")
            SR_nparray = SR_nparray.astype(np.float32) / 255.
            # Convert RGB format to ycbcr format
            SR_YCbCr_nparray = Utils.img_rgb2ycbcr(SR_nparray, sep_y_channel=True)
            SR_YCbCr_tensor = F.to_tensor(SR_YCbCr_nparray).to(device).unsqueeze_(0)

            # Pytorch tensor to numpy array image
            HR_nparray = HR_imgs.squeeze_(0).permute(1, 2, 0).mul_(255).clamp_(0, 255).cpu().numpy().astype("uint8")
            HR_nparray = HR_nparray.astype(np.float32) / 255.
            # Convert RGB format to ycbcr format
            HR_YCbCr_nparray = Utils.img_rgb2ycbcr(HR_nparray, sep_y_channel=True)
            HR_YCbCr_tensor = F.to_tensor(HR_YCbCr_nparray).to(device).unsqueeze_(0)

            # measure accuracy and record loss
            psnr = 10. * torch.log10(1. / psnr_criterion(SR_YCbCr_tensor, HR_YCbCr_tensor))
            avg_meter_PSNR.update(psnr.item(), LR_imgs.size(0))

            # Record training log information

            if batch_idx % log_freq == 0:
                progress.display(batch_idx)

    if mode == "valid":
        writer.add_scalar("Valid/PSNR", avg_meter_PSNR.avg, epoch + 1)
    elif mode == "test":
        writer.add_scalar("Test/PSNR", avg_meter_PSNR.avg, epoch + 1)

    return avg_meter_PSNR.avg


# ----------------------------------------------------------------------------------
# Helper functions for model training visualization from
# "https://github.com/pytorch/examples/blob/master/imagenet/main.py"
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.2f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.2f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.2f}"
        else:
            raise ValueError(f"Invalid summary type {self.summary_type}")

        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print("\t".join(entries))

    def display_summary(self):
        entries = [" *"]
        entries += [meter.summary() for meter in self.meters]
        print(" ".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


if __name__ == '__main__':
    torch.cuda.set_device(2)  # Choose the number of GPU on which we train the model.
    # ---------------------------------------------------------------------------------------------------
    # Get params from command lines.
    p = argparse.ArgumentParser()
    p.add_argument("--track", type=str)
    # p.add_argument("--epochNum", default=200, type=int)
    args = p.parse_args()

    # --------------------------------------------------------------------------------------------------
    # Create the dataset paths of different tracks
    train_LR_track_paths = ["Datasets/train/LR_bicubic_X2", "Datasets/train/LR_bicubic_X3",
                            "Datasets/train/LR_bicubic_X4",
                            "Datasets/train/LR_unknown_X2", "Datasets/train/LR_unknown_X3",
                            "Datasets/train/LR_unknown_X4"]

    valid_LR_track_paths = ["Datasets/valid/LR_bicubic_X2", "Datasets/valid/LR_bicubic_X3",
                            "Datasets/valid/LR_bicubic_X4",
                            "Datasets/valid/LR_unknown_X2", "Datasets/valid/LR_unknown_X3",
                            "Datasets/valid/LR_unknown_X4"]

    test_LR_track_paths = ["Datasets/test/LR_bicubic_X2", "Datasets/test/LR_bicubic_X3", "Datasets/test/LR_bicubic_X4",
                           "Datasets/test/LR_unknown_X2", "Datasets/test/LR_unknown_X3", "Datasets/test/LR_unknown_X4"]

    # Create a dictionary for mapping of the command and the indices of the dataset folder paths.
    dic_dataset_path = {
        "BicubicX2":0,
        "BicubicX3":1,
        "BicubicX4":2,
        "UnknownX2":3,
        "UnknownX3":4,
        "UnknownX4":5
    }
    index = dic_dataset_path[args.track]
    LR_train_folder_path = train_LR_track_paths[index]
    LR_valid_folder_path = valid_LR_track_paths[index]
    LR_test_folder_path = test_LR_track_paths[index]

    # Create a dictionary to map the command with the upscale factor value.
    dic_upscale_factor = {
        "BicubicX2": 2,
        "BicubicX3": 3,
        "BicubicX4": 4,
        "UnknownX2": 2,
        "UnknownX3": 3,
        "UnknownX4": 4
    }
    upscale_factor = dic_upscale_factor[args.track]
    train_GAN_model(LR_train_folder_path, LR_valid_folder_path, LR_test_folder_path, upscale_factor, args.track)
