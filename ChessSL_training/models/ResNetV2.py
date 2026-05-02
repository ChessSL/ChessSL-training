import torch.nn.functional as F
import torch.nn as nn

from .az_policy_map import AZPolicyMap


class ResNetV2(nn.Module):
    """ResNet with pre-activation."""

    def __init__(
        self,
        num_filters: int,
        num_blocks: int,
        mlh_channels: int,
        mlh_fc_size: int,
    ):
        super().__init__()
        self.num_filters = num_filters
        self.num_blocks = num_blocks
        self.mlh_channels = mlh_channels
        self.mlh_fc_size = mlh_fc_size

        self.conv1 = nn.Conv2d(
            in_channels=112,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.blocks = nn.ModuleList(
            [ResidualBlockV2(num_filters) for _ in range(num_blocks)]
        )
        self.bn_out = nn.BatchNorm2d(num_filters)

        self.policy_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=80,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            AZPolicyMap(),
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=32,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )

        self.moves_left_head = nn.Sequential(
            nn.Conv2d(
                in_channels=num_filters,
                out_channels=mlh_channels,
                kernel_size=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(mlh_channels),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(mlh_channels * 8 * 8, self.mlh_fc_size),
            nn.ReLU(),
            nn.Linear(mlh_fc_size, 1),
            nn.ReLU(),
        )

    def forward(self, x):
        out = self.conv1(x)
        for block in self.blocks:
            out = block(out)
        out = self.bn_out(out)
        out = F.relu(out)
        policy = self.policy_head(out)
        value = self.value_head(out)
        moves_left = self.moves_left_head(out)
        return policy, value, moves_left


class ResidualBlockV2(nn.Module):
    """A ResNet block with pre-activation."""

    def __init__(self, num_filters: int):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(num_filters)
        self.conv1 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm2d(num_filters)
        self.conv2 = nn.Conv2d(
            in_channels=num_filters,
            out_channels=num_filters,
            kernel_size=3,
            padding=1,
            bias=False,
        )

    def forward(self, x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = out + x

        return out
