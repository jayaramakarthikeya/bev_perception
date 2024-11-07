import torch
import torch.nn as nn
import warnings

class SECOND(nn.Module):
    """Backbone network for SECOND/PointPillars/PartA2/MVXNet.

    Args:
        in_channels (int): Input channels.
        out_channels (list[int]): Output channels for multi-scale feature maps.
        layer_nums (list[int]): Number of layers in each stage.
        layer_strides (list[int]): Strides of each stage.
        norm_cfg (dict): Config dict of normalization layers.
        conv_cfg (dict): Config dict of convolutional layers.
    """

    def __init__(
        self,
        in_channels=128,
        out_channels=[128, 128, 256],
        layer_nums=[3, 5, 5],
        layer_strides=[2, 2, 2],
        norm_cfg=dict(type="BN", eps=1e-3, momentum=0.01),
        conv_cfg=dict(type="Conv2d", bias=False),
        init_cfg=None,
        pretrained=None,
    ):
        super().__init__()
        assert len(layer_strides) == len(layer_nums)
        assert len(out_channels) == len(layer_nums)

        in_filters = [in_channels, *out_channels[:-1]]
        blocks = []
        for i, layer_num in enumerate(layer_nums):
            block = [
                nn.Conv2d(
                    in_filters[i],
                    out_channels[i],
                    kernel_size=3,
                    stride=layer_strides[i],
                    padding=1,
                    bias=conv_cfg.get("bias", False)
                ),
                nn.BatchNorm2d(out_channels[i], eps=norm_cfg.get("eps", 1e-3), momentum=norm_cfg.get("momentum", 0.01)),
                nn.ReLU(inplace=True),
            ]
            for j in range(layer_num):
                block.append(
                    nn.Conv2d(
                        out_channels[i],
                        out_channels[i],
                        kernel_size=3,
                        padding=1,
                        bias=conv_cfg.get("bias", False)
                    )
                )
                block.append(nn.BatchNorm2d(out_channels[i], eps=norm_cfg.get("eps", 1e-3), momentum=norm_cfg.get("momentum", 0.01)))
                block.append(nn.ReLU(inplace=True))

            block = nn.Sequential(*block)
            blocks.append(block)

        self.blocks = nn.ModuleList(blocks)

        if isinstance(pretrained, str):
            warnings.warn(
                "DeprecationWarning: pretrained is deprecated, "
                'please use "init_cfg" instead'
            )
            self.init_cfg = dict(type="Pretrained", checkpoint=pretrained)
        else:
            self.init_cfg = dict(type="Kaiming", layer="Conv2d")

    def forward(self, x):
        """Forward function.

        Args:
            x (torch.Tensor): Input with shape (N, C, H, W).

        Returns:
            tuple[torch.Tensor]: Multi-scale features.
        """
        outs = []
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
            outs.append(x)
        return tuple(outs)
