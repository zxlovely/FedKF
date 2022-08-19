from torch import nn


class LeNet5(nn.Module):
    def __init__(self, img_channel=1, num_classes=10):
        super(LeNet5, self).__init__()
        self.name = 'LeNet5'
        self.backbone = nn.Sequential(
            nn.Conv2d(img_channel, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 120, kernel_size=5),
            nn.ReLU(inplace=True)
        )
        self.classifier = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.backbone(x)
        features = x.flatten(start_dim=1)
        logits = self.classifier(features)

        outputs = {}
        outputs['features'] = features
        outputs['logits'] = logits

        return outputs
