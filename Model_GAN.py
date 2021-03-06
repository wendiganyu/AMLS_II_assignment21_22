import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class ResidualBlock(nn.Module):
    """
    Define the residual convolutional residual block.
    """

    def __init__(self, channel_num):
        """
        Initialize.
        :param channel_num: The input channel number and the output number of a Conv2d are set as the same.
        """
        super(ResidualBlock, self).__init__()
        self.blk = nn.Sequential(
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num),
            nn.PReLU(),
            nn.Conv2d(channel_num, channel_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channel_num)
        )

    def forward(self, input_x):
        identity_map = input_x

        out = self.blk(input_x)
        out = torch.add(out, identity_map)

        return out

class UpSampleBlk(nn.Module):
    def __init__(self, input_channels, upscale_factor):
        super(UpSampleBlk,self).__init__()
        if upscale_factor == 3:
            PixelShuffle_param = 3
        else: # upscale_factor == 2 or upscale_factor == 4
            PixelShuffle_param = 2
        self.up_sample_blk = nn.Sequential(
            nn.Conv2d(input_channels, input_channels * PixelShuffle_param * PixelShuffle_param, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(PixelShuffle_param),
            nn.PReLU()
        )

    def forward(self, input_x):
        out = self.up_sample_blk(input_x)

        return out


class Generator(nn.Module):
    def __init__(self, upscale_factor):
        super(Generator,self).__init__()

        # First convolutional block
        self.conv_blk1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding= 4),
            nn.PReLU()
        )

        # A sequence of residual blocks.
        residual_blk_sequence = []
        for _ in range(16):
            residual_blk_sequence.append(ResidualBlock(64))
        self.residual_blk_sequence = nn.Sequential(*residual_blk_sequence)

        # Second Conv block
        self.conv_blk2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64)
        )

        # Upsample block sequence
        upsample_blk_seq = []

        # If upscale_factor ==2 or upscale_factor == 3, use only one unsampling block.
        if upscale_factor == 4:
            iter = 2
        else:
            iter = 1

        for _ in range(iter):
            upsample_blk_seq.append(UpSampleBlk(input_channels=64, upscale_factor=upscale_factor))
        self.upsample_blk_seq = nn.Sequential(*upsample_blk_seq)

        #Output
        self.conv_blk3 = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

        # self._initialize_weights()
    def forward(self, input_x):
        Y1 = self.conv_blk1(input_x)
        Y = self.residual_blk_sequence(Y1)
        Y2 = self.conv_blk2(Y)
        Y = torch.add(Y1, Y2)
        Y = self.upsample_blk_seq(Y)
        Y = self.conv_blk3(Y)

        return Y

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.feature_extractor = nn.Sequential(
            # input size: 3 * 180 * 180
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),

            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1), # Now:64 * 90 * 90
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), # Now: 128 * 90 * 90
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),

            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), # Now: 128 * 45 * 45
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), # Now: 256 * 45 * 45
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # Now: 256 * 23 * 23
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), # Now : 512 * 23 * 23
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),

            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1), # Now: 512 * 12 * 12
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, True)
            # nn.PReLU()

        )

        self.classifier = nn.Sequential(
            nn.Linear(512*12*12, 1024),
            nn.LeakyReLU(0.2, True),
            # nn.PReLU(),
            nn.Linear(1024, 1)

        )

    def forward(self, input_x):
        Y = self.feature_extractor(input_x)
        Y = torch.flatten(Y, 1)
        Y = self.classifier(Y)

        return Y

class ContentLoss(nn.Module):
    """
    Content loss calculation based on a VGG19 network already trained on ImageNet.
    """

    def __init__(self):
        super(ContentLoss, self).__init__()
        # Load VGG19 trained on ImageNet.
        vgg19 = models.vgg19(pretrained=True).eval()
        # Extract 36-th layer's output of vgg19 for content loss.
        self.feature_extractor = nn.Sequential(*list(vgg19.features.children())[:36])

        for parameter in self.feature_extractor.parameters():
            parameter.requires_grad = False

        # The constant values of mean and the standard are the parameters of VGG19 used on ImageNet.
        # self.register_buffer("mean", torch.Tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        # self.register_buffer("std", torch.Tensor([0.229, 0.224, 0.225]).view(1,3,1,1))

    def forward(self, super_res_img, high_res_img):
        # Normalization
        # super_res_img = super_res_img.sub(self.mean).div(self.std)
        # high_res_img = high_res_img.sub(self.mean).div(self.std)

        # Calculate the content loss
        loss = F.mse_loss(self.feature_extractor(super_res_img), self.feature_extractor(high_res_img))

        return loss