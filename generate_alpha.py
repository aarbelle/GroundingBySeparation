import numpy as np
from perlin_noise import generate_perlin_noise_2d
import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).

    Usage Example:
    smoothing = GaussianSmoothing(3, 5, 1)
    input = torch.rand(1, 3, 100, 100)
    input = F.pad(input, (2, 2, 2, 2), mode='reflect')
    output = smoothing(input)
    """

    def __init__(self, channels, kernel_size, sigma, dim=2):
        super(GaussianSmoothing, self).__init__()
        if isinstance(kernel_size, numbers.Number):
            kernel_size = [kernel_size] * dim
        if isinstance(sigma, numbers.Number):
            sigma = [sigma] * dim

        # The gaussian kernel is the product of the
        # gaussian function of each dimension.
        kernel = 1
        meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in kernel_size
            ]
        )
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)

        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

        # Reshape to depthwise convolutional weight
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, x):
        """
        Apply gaussian filter to input.
        Arguments:
            x (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(x, weight=self.weight, groups=self.groups)


def sig_and_strech(x, t):
    y = np.clip((x / t), -100, 100)
    sig = 1 - 1 / (1 + np.exp(y))
    sig -= sig.min()
    sig /= sig.max()
    return sig


def get_perlin_alpha(height, width, alpha_downscale, rng, seeds, masks):
    alphas = []
    y, x = height // alpha_downscale, width // alpha_downscale
    for mask in masks:
        nonz = mask[0].nonzero(as_tuple=False)
        p0, _ = nonz.min(dim=0)
        p1, _ = nonz.max(dim=0)
        y0, x0 = p0.numpy() // alpha_downscale
        y1, x1 = (p1.numpy() + 1) // alpha_downscale
        ya = (y1 - y0)
        xa = (x1 - x0)

        # noinspection PyArgumentList
        temp = rng.randn() / 10
        alpha = 1.
        for s in seeds:
            a = generate_perlin_noise_2d((ya, xa), (s+1, s+1), rng)
            a = sig_and_strech(a, temp)
            a = a - a.min()
            a = a / a.max()
            a = a.clip(0, 1)
            alpha *= a
        pad_alpha = np.zeros((y, x), dtype=np.float32)
        pad_alpha[y0:y1, x0:x1] = alpha
        alphas.append(torch.from_numpy(pad_alpha))
    alpha_ = torch.stack(alphas, 0).unsqueeze(1)
    alpha = torch.nn.functional.interpolate(alpha_, (height, width), mode='bilinear', align_corners=False)
    return alpha


def get_gmm_alpha(batch_size, height, width, num_seeds, rng=None):
    if rng is None:
        rng = np.random.RandomState()
    y_grid, x_grid = torch.meshgrid(torch.arange(height), torch.arange(width))
    y_grid = y_grid.unsqueeze(0)
    x_grid = x_grid.unsqueeze(0)
    alpha = None
    mixture = torch.from_numpy(rng.rand(num_seeds, batch_size, 1, 1)).float()
    mixture = mixture/mixture.sum(dim=0)
    for mix in mixture:
        mu_x = torch.from_numpy(rng.randint(width, size=[batch_size, 1, 1])).float()
        mu_y = torch.from_numpy(rng.randint(height, size=[batch_size, 1, 1])).float()
        sig_x = torch.from_numpy(rng.randint(width//32, width//4, size=[batch_size, 1, 1])).float()
        sig_y = torch.from_numpy(rng.randint(height//32, height//4, size=[batch_size, 1, 1])).float()
        cent_x = x_grid-mu_x
        cent_y = y_grid-mu_y
        md = torch.square(cent_x.float()/sig_x.float()) + torch.square(cent_y.float()/sig_y.float())
        gauss = torch.exp(-0.5*md)
        if alpha is None:
            alpha = mix*gauss
        else:
            alpha += mix*gauss

    alpha = alpha.clamp(0, 1)
    alpha = alpha.unsqueeze(1)
    return alpha


