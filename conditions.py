import torch
from torch.nn.functional import normalize


def dist_cond(x: torch.Tensor, c: torch.Tensor, **kwargs):
    """

    :param x: the feature vector of the Unet. shape (N,F,H,W)
    :param c: the external condition feature vectors. shape (C,N,F)
    :return: (C,N,F,H,W)
    """
    x_orig = x
    # x = normalize(x, p=2, dim=1)
    # c = normalize(c, p=2, dim=2)

    c = c.unsqueeze(3).unsqueeze(4)  # C, N, F, 1, 1
    x = x.unsqueeze(0)  # 1, N, F, H, W
    if torch.isinf(x).any():
        print('Found inf in X')
    if torch.isnan(x).any():
        print('Found nan in X')
    distance = torch.abs(x - c)  # C, N, F, H, W (C=2, N=1, F=1, H=W=1)
    with torch.no_grad():
        valid_c = torch.all(torch.eq(c, 0), dim=2, keepdim=True).float()
    distance = distance + (100*distance.detach().max()*valid_c.detach())
    distance.clamp(0., 100)
    atten = torch.exp(-distance)

    return atten * x, atten


class BaseCond(torch.nn.Module):
    def __init__(self, cap_in_depth=None, out_depth=None, im_in_depth=None, **kwargs):
        super(BaseCond, self).__init__()
        self.kwargs = kwargs
        if cap_in_depth is not None:
            self.cap_fc = torch.nn.Linear(cap_in_depth, out_depth)
        else:
            self.cap_fc = None
        self.attn = None
        if im_in_depth is not None:
            self.im_fc = torch.nn.Sequential(torch.nn.Conv2d(im_in_depth, out_depth, 1),  torch.nn.BatchNorm2d(out_depth),
                                             torch.nn.ReLU())
        else:
            self.im_fc = None


class DistCond(BaseCond):
    def __init__(self, *args, **kwargs):
        self.attn = None
        self.cap_features = None
        super(DistCond, self).__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        x, c = inputs

        with torch.no_grad():
            valid_c = 1 - torch.all(torch.eq(c, 0), dim=2, keepdim=True).float()
        c = torch.stack([self.cap_fc(cc) for cc in c]) * valid_c
        self.cap_features = c
        if self.im_fc is not None:
            x = self.im_fc(x)
        dist, attn = dist_cond(x, c, **kwargs)
        self.attn = attn
        return dist


class NoCond(BaseCond):
    def __init__(self, *args, **kwargs):
        super(NoCond, self).__init__(*args, **kwargs)

    def forward(self, *inputs, **kwargs):
        x, _ = inputs
        if self.im_fc is not None:
            x = self.im_fc(x)
        x = normalize(x, p=2, dim=1)
        if len(x.shape) == 3:
            x.unsqueeze(0)
        return x
