import math
from torch import nn


def conv(in_planes, out_planes, kernel_size, stride, norm=None):
    if norm == 'bn':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(out_planes)
        )
    if norm == 'gn':
        return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            nn.GroupNorm(num_groups=out_planes, num_channels=out_planes)
        )

    return nn.Conv2d(in_planes, out_planes, kernel_size, stride=stride, padding=kernel_size // 2)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride, norm=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv(in_planes, planes, 3, stride, norm)
        self.conv2 = conv(planes, planes * self.expansion, 3, 1, norm)
        self.relu = nn.ReLU(inplace=True)
        self.down_sample = None
        if stride != 1 or in_planes != planes * self.expansion:
            self.down_sample = conv(in_planes, planes * self.expansion, 1, stride, norm)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)

        if self.down_sample is not None:
            residual = self.down_sample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNetCIFAR(nn.Module):
    def __init__(self, name, block, nums_blocks, num_classes, norm=None):
        super(ResNetCIFAR, self).__init__()
        self.name = name
        self.conv1 = conv(3, 64, 3, 1, norm)
        self.relu = nn.ReLU(inplace=True)
        self.in_planes = 64
        self.layer1 = self._make_layer(block, 64, nums_blocks[0], 1, norm)
        self.layer2 = self._make_layer(block, 128, nums_blocks[1], 2, norm)
        self.layer3 = self._make_layer(block, 256, nums_blocks[2], 2, norm)
        self.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        self.classifier = nn.Linear(in_features=self.in_planes, out_features=num_classes)

        self._weight_initialization()

    def _make_layer(self, block, planes, num_blocks, stride, norm):
        blocks = []
        for _ in range(num_blocks):
            blocks.append(block(self.in_planes, planes, stride, norm))
            self.in_planes = planes * block.expansion
            stride = 1

        return nn.Sequential(*blocks)

    def _weight_initialization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg_pool(x)
        features = x.flatten(start_dim=1)
        logits = self.classifier(features)

        outputs = {}
        outputs['features'] = features
        outputs['logits'] = logits

        return outputs


def resnet8(num_classes=10, norm=None):
    return ResNetCIFAR('ResNet8', BasicBlock, [1, 1, 1], num_classes, norm)
