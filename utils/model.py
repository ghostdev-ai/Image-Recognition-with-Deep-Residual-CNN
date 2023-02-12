import torch
from torch import nn

class ResNet_18_Layer(nn.Module):
    """Creates the 18-layer ResNet architecture.

  Replicates the 18-layer ResNet architecture from the 
  paper Deep Residual Learning for Image Recognition in PyTorch.

  See the paper here: https://arxiv.org/abs/1512.03385

  Args:
    input_shape: An integer indicating number of input channels.
    hidden_units: An integer indicating number of hidden units between layers.
    output_shape: An integer indicating number of output units.
  """
    def __init__(self) -> None:
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            ResBlock(hidden_shape=64),
            ResBlock(hidden_shape=64))
        
        self.conv3_1 = ConvBlock(input_shape=64, output_shape=128, pool=True)
        
        self.conv3_x = nn.Sequential(
            ResBlock(hidden_shape=128),
            ResBlock(hidden_shape=128))
        
        self.conv4_1 = ConvBlock(input_shape=128, output_shape=256, pool=True)
        
        self.conv4_x = nn.Sequential(
            ResBlock(hidden_shape=256),
            ResBlock(hidden_shape=256))
        
        self.conv5_1 = ConvBlock(input_shape=256, output_shape=512, pool=True)
        
        self.conv5_x = nn.Sequential(
            ResBlock(hidden_shape=512),
            ResBlock(hidden_shape=512))

        self.classifier = nn.Sequential(
            nn.AvgPool2d(kernel_size=7, stride=7, padding=0),
            nn.Flatten(start_dim=1, end_dim=-1),
            nn.Linear(in_features=512, out_features=1000, bias=True)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2_x(x)
        x = self.conv3_x( self.conv3_1(x) )
        x = self.conv4_x( self.conv4_1(x) )
        x = self.conv5_x( self.conv5_1(x) )
        return self.classifier(x)


class ResBlock(nn.Module):
    def __init__(self, hidden_shape, stride=1):
        super().__init__()

        self.layers = nn.Sequential(
            ConvBlock(hidden_shape, hidden_shape),                            
            ConvBlock(hidden_shape, hidden_shape)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x) + x


class ConvBlock(nn.Module):
    def __init__(self, input_shape: int, output_shape: int, stride: int=1, pool: bool=False):
        super().__init__()

        layers = [nn.Conv2d(input_shape, output_shape, kernel_size=3, padding=1),
              nn.BatchNorm2d(output_shape),
              nn.ReLU(inplace=True)]  # inplace=True - modifies the input tensor directly without the allocation of additional memory.
        if pool: layers.append(nn.MaxPool2d(2))
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)