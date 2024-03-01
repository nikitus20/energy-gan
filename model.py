import torch
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

class EnergyDistanceDiscriminator(nn.Module):
    def __init__(self, channels_img):
        super(EnergyDistanceDiscriminator, self).__init__()
        # No learnable parameters needed for pure energy distance calculation

    def forward(self, real_imgs, gen_imgs):
        # Calculate pairwise distances within real and within generated images
        real_real_dist = self.pairwise_distances(real_imgs, real_imgs)
        gen_gen_dist = self.pairwise_distances(gen_imgs, gen_imgs)

        # Calculate pairwise distances between real and generated images
        real_gen_dist = self.pairwise_distances(real_imgs, gen_imgs)

        # Compute energy distance
        energy_distance = 2 * real_gen_dist.mean() - real_real_dist.mean() - gen_gen_dist.mean()
        return energy_distance

    def pairwise_distances(self, x, y):
        # Compute pairwise distances between two sets of images
        # x: tensor of shape [m, c, h, w]
        # y: tensor of shape [n, c, h, w]
        x_flat = x.view(x.size(0), -1)
        y_flat = y.view(y.size(0), -1)

        distances = torch.cdist(x_flat, y_flat, p=2)  # Euclidean distance (L2 norm)
        return distances


class Discriminator(nn.Module):
    def __init__(self, channels_img, features_d):
        super(Discriminator, self).__init__()
        self.disc = nn.Sequential(
            # input: N x channels_img x 64 x 64
            nn.Conv2d(
                channels_img, features_d, kernel_size=4, stride=2, padding=1
            ),
            nn.LeakyReLU(0.2),
            # _block(in_channels, out_channels, kernel_size, stride, padding)
            self._block(features_d, features_d * 2, 4, 2, 1),
            self._block(features_d * 2, features_d * 4, 4, 2, 1),
            self._block(features_d * 4, features_d * 8, 4, 2, 1),
            # After all _block img output is 4x4 (Conv2d below makes into 1x1)
            nn.Conv2d(features_d * 8, 1, kernel_size=4, stride=2, padding=0),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
        )

    def forward(self, x):
        return self.disc(x)


class Generator(nn.Module):
    def __init__(self, channels_noise, channels_img, features_g):
        super(Generator, self).__init__()
        self.net = nn.Sequential(
            # Input: N x channels_noise x 1 x 1
            self._block(channels_noise, features_g * 16, 4, 1, 0),  # img: 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # img: 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # img: 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # img: 32x32
            nn.ConvTranspose2d(
                features_g * 2, channels_img, kernel_size=4, stride=2, padding=1
            ),
            # Output: N x channels_img x 64 x 64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


def initialize_weights(model):
    # Initializes weights according to the DCGAN paper
    for m in model.modules():
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.BatchNorm2d)):
            nn.init.normal_(m.weight.data, 0.0, 0.02)


def test():
    N, in_channels, H, W = 8, 3, 64, 64
    noise_dim = 100
    x = torch.randn((N, in_channels, H, W))
    gen = Generator(noise_dim, in_channels, 8)
    z = torch.randn((N, noise_dim, 1, 1))
    assert gen(z).shape == (N, in_channels, H, W), "Generator test failed"
    disc = EnergyDistanceDiscriminator(in_channels)
    #assert disc(x, gen(z)).shape == (1), "Discriminator test failed"
   
    print("Success, tests passed!")

if __name__ == "__main__":
    test()