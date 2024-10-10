from torchvision import datasets, transforms
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

class SDFNet(nn.Module):
    """ 2D Neural SDF Network
    """
    def __init__(self,
                latent_size,
                latent_in=[3],
                dims=[64, 64, 64, 64, 64, 64],
                dropout=(0, 1, 2, 3, 4, 5),
                dropout_prob=0.2,
                norm_layers=(0, 1, 2, 3, 4, 5),
                weight_norm=True,
                latent_dropout=False,
                **kwargs
                ):
        super(SDFNet, self).__init__()
        
        self.latent_size = latent_size
        self.latent_dropout = latent_dropout
        self.latent_in = latent_in
        self.norm_layers = norm_layers
        self.weight_norm = weight_norm 
        dims = [latent_size + 2] + dims + [1]

        self.fcn = []
        self.num_layers = len(dims)
        for l in range(self.num_layers-1):
            if l + 1 in latent_in:
                out_dim = dims[l+1] - (latent_size + 2)
            else:
                out_dim = dims[l+1]

            if weight_norm and l in self.norm_layers:
                self.fcn += [nn.utils.weight_norm(nn.Linear(dims[l], out_dim))]
            else:
                self.fcn += [nn.Linear(dims[l], out_dim)]

            # batchnorm if norm, but not weight norm. It wont happen
            if (not(weight_norm) and self.norm_layers is not None and l in self.norm_layers):
                setattr(self, "bn" + str(l), nn.LayerNorm(out_dim))
        
        # self.fcn = nn.Sequential( *self.fcn)
        self.fcn = nn.ModuleList(self.fcn)
        
        # Dropout
        self.dropout = dropout
        self.dropout_prob = dropout_prob
        # Tanh to put values between -1 and 1
        self.th = nn.Tanh()
        self.relu = nn.ReLU()
        self.do = nn.Dropout(p=self.dropout_prob)
        
    
    def forward(self, input):
        xy = input[:, -3:] 
        latent = input[:, :-3]

        # Apply dropout to latent vector during training
        if input.shape[1] > 2 and self.latent_dropout:
            latent = F.dropout(latent, p=0.2, training=self.training)
            x = torch.cat([latent, xy], 1)
        else:
            x = input

        # traverse linear and norm layers
        for l in range(self.num_layers-2):
            # inject latent code and coordinates back into image
            if l in self.latent_in:
                x = torch.cat([x, input], 1)

            # linear pass
            x = self.fcn[l](x)
            # if batch norm
            if (
                    self.norm_layers is not None
                    and l in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(l))
                    x = bn(x)
            # relu
            x = self.relu(x)
            if self.dropout is not None and l in self.dropout:
                x = self.do(x)

        # last linear layer pass
        s = self.fcn[-1](x)
        # s = self.th(x)
        return s

class Decoder(nn.Module):
    """ Original DeepSDF Decoder
    """
    def __init__(
        self,
        latent_size,
        dims,
        dropout=None,
        dropout_prob=0.0,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        xyz_in_all=None,
        use_tanh=False,
        latent_dropout=False,
    ):
        super(Decoder, self).__init__()

        def make_sequence():
            return []

        dims = [latent_size + 2] + dims + [1]

        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 2

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        self.th = nn.Tanh()

    # input: N x (L+3)
    def forward(self, input):
        xyz = input[:, -2:]

        if input.shape[1] > 2 and self.latent_dropout:
            latent_vecs = input[:, :-2]
            latent_vecs = F.dropout(latent_vecs, p=0.2, training=self.training)
            x = torch.cat([latent_vecs, xyz], 1)
        else:
            x = input

        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            # last layer Tanh
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x
    
