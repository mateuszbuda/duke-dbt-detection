import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from torchvision.models.densenet import _DenseBlock, _Transition


class DenseYOLO(nn.Module):

    def __init__(
        self,
        img_channels,
        out_channels,
        growth_rate=16,
        block_config=(2, 6, 4, 12, 8),
        num_init_features=8,
        bn_size=4,
        drop_rate=0.0,
    ):
        super(DenseYOLO, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            in_channels=img_channels,
                            out_channels=num_init_features,
                            kernel_size=5,
                            padding=2,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_features=num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=2, stride=2)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module("norm1", nn.BatchNorm2d(num_features))

        self.features.add_module(
            "conv1",
            nn.Conv2d(
                in_channels=num_features,
                out_channels=out_channels,
                kernel_size=3,
                stride=3,
                bias=False,
            ),
        )

        # initialization
        p = 1.0 / 77.0  # prior for output assumes 1 box per grid of size 11x7
        b = -1.0 * np.log10((1.0 - p) / p)  # bias for output layer based on focal loss paper
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                if name == "features.norm1":
                    nn.init.constant_(module.bias, b)
                else:
                    nn.init.constant_(module.bias, 0)

    def forward(self, x):
        x = self.features(x)
        obj = torch.sigmoid(x[:, [0]].clone())
        loc = torch.tanh(x[:, [1, 2]].clone())
        box = torch.sigmoid(x[:, [3, 4]].clone())
        x = torch.cat((obj, loc, box), dim=1)
        return x
