from torch import nn
import torch


class ResBlock(nn.Module):
    def __init__(self,
                 input_shape: int,  # 64
                 output_shape: int,  # 128
                 stride: int = 1,  # 2 -> downsample
                 downsample=None
                 ) -> None:
        super().__init__()

        self.conv_block = nn.Sequential(
            # in_features = 64, out_features = 128, stride = 2
            nn.Conv2d(input_shape,
                      output_shape,
                      kernel_size=3,
                      padding=1,
                      stride=stride),
            nn.BatchNorm2d(output_shape),
            nn.ReLU(inplace=True),

            nn.Conv2d(output_shape,
                      output_shape,
                      kernel_size=3,
                      padding=1),
            nn.BatchNorm2d(output_shape))

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv_block(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class ResNet18(nn.Module):
    """
    Creates the 18-layer ResNet architecture.

    Replicates the 18-layer ResNet architecture from the 
    paper Deep Residual Learning for Image Recognition in PyTorch.

    See the paper here: https://arxiv.org/abs/1512.03385

    Args:
      input_shape: An integer indicating number of color channels.
      output_shape: An integer indicating number of output classes.
    """

    def __init__(self,
                 input_shape: int,
                 output_shape: int
                 ) -> None:
        super().__init__()
        # Output Size: 112 x 112
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))  # inplace=True - modifies the input tensor directly without the allocation of additional memory.
        # Output Size: 56 X 56
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),
            ResBlock(input_shape=64,
                     output_shape=64),
            ResBlock(input_shape=64,
                     output_shape=64))
        # Output Size: 28 X 28
        self.conv3_x = nn.Sequential(
            ResBlock(input_shape=64,
                     output_shape=128,
                     stride=2,
                     downsample=self.downsample(input_shape=64,
                                                output_shape=128)),
            ResBlock(input_shape=128,
                     output_shape=128))
        # Output Size: 14 X 14
        self.conv4_x = nn.Sequential(
            ResBlock(input_shape=128,
                     output_shape=256,
                     stride=2,
                     downsample=self.downsample(input_shape=128,
                                                output_shape=256)),
            ResBlock(input_shape=256,
                     output_shape=256))
        # Output Size: 7 X 7
        self.conv5_x = nn.Sequential(
            ResBlock(input_shape=256,
                     output_shape=512,
                     stride=2,
                     downsample=self.downsample(input_shape=256,
                                                output_shape=512)),
            ResBlock(input_shape=512,
                     output_shape=512))
        # Output Size: 1 X 1
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512,
                            out_features=output_shape)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # print(f"Output Size [X]: {x.shape}")
        out = self.conv1(x)
        # print(f"Output Size [7x7 conv, 64, /2]: {out.shape}")
        out = self.conv2_x(out)
        # print(f"Output Size [conv2_x]: {out.shape}")
        out = self.conv3_x(out)
        # print(f"Output Size [conv3_x]: {out.shape}")
        out = self.conv4_x(out)
        # print(f"Output Size [conv4_x]: {out.shape}")
        out = self.conv5_x(out)
        # print(f"Output Size [conv5_x]: {out.shape}")
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        # print(f"Output Size [average pool, 1000-d fc, softmax]: {out}")
        return out


    def downsample(self, input_shape: int, output_shape: int):
        return nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=output_shape))


class ResNet34(nn.Module):
    def __init__(self,
                 input_shape: int,
                 output_shape: int
                 ) -> None:
        super().__init__()
        # Output Size: 112 x 112
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=64,
                      kernel_size=7,
                      stride=2,
                      padding=3)
        )
        # Output Size: 112 x 112
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,
                         stride=2,
                         padding=1),
            ResBlock(input_shape=64,
                     output_shape=64),
            ResBlock(input_shape=64,
                     output_shape=64),
            ResBlock(input_shape=64,
                     output_shape=64))
        # Output Size: 112 x 112
        self.conv3_x = nn.Sequential(
            ResBlock(input_shape=64,
                     output_shape=128,
                     stride=2,
                     downsample=self.downsample(input_shape=64,
                                                 output_shape=128)),
            ResBlock(input_shape=128,
                     output_shape=128),
            ResBlock(input_shape=128,
                     output_shape=128),
            ResBlock(input_shape=128,
                     output_shape=128))
        # Output Size: 112 x 112
        self.conv4_x = nn.Sequential(
            ResBlock(input_shape=128,
                     output_shape=256,
                     stride=2,
                     downsample=self.downsample(input_shape=128,
                                                 output_shape=256)),
            ResBlock(input_shape=256,
                     output_shape=256),
            ResBlock(input_shape=256,
                     output_shape=256),
            ResBlock(input_shape=256,
                     output_shape=256),
            ResBlock(input_shape=256,
                     output_shape=256),
            ResBlock(input_shape=256,
                     output_shape=256))
        # Output Size: 112 x 112
        self.conv5_x = nn.Sequential(
            ResBlock(input_shape=256,
                     output_shape=512,
                     stride=2,
                     downsample=self.downsample(input_shape=256,
                                                 output_shape=512)),
            ResBlock(input_shape=512,
                     output_shape=512),
            ResBlock(input_shape=512,
                     output_shape=512))
        # Output Size: 112 x 112
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.fc = nn.Linear(in_features=512,
                             out_features=output_shape)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgpool(out)
        out = torch.flatten(out, start_dim=1)
        out = self.fc(out)
        return out


    def downsample(self, input_shape: int, output_shape: int) -> torch.Tensor:
        return nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=output_shape,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.BatchNorm2d(num_features=output_shape))
