import torch.nn as nn
from torchvision.models import resnet18, resnet34, resnet50, vgg19, vgg16

models = {
    "resnet18": resnet18,
    "resnet34": resnet34,
    "resnet50": resnet50,
    "vgg19": vgg19,
    "vgg16": vgg16
}


class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.3)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(0.3)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Dropout2d(0.3)
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(128 * 12 * 12, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            # nn.Dropout(0.5),
            nn.Linear(256, 2),
        )
        return

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.linear_layers(out)
        return out


class BinaryClassifier(nn.Module):
    def __init__(self, net_name="resnet18"):
        super().__init__()
        if net_name not in models:
            raise ValueError(f"Model {net_name} has not been adapted to binary classification.")
        print(f"Using {net_name} model not pretrained on Imagenet")
        self.net = models[net_name]()
        if "resnet" in net_name:
            assert isinstance(self.net.fc, nn.Linear), "Last layer is not linear. Pytorch code may have changed"
            self.net.fc = nn.Linear(self.net.fc.in_features, 2, self.net.fc.bias is not None)
        elif "vgg" in net_name:
            assert isinstance(self.net.classifier[-1],
                              nn.Linear), "Last layer is not linear. Pytorch code may have changed"
            self.net.classifier[-1] = nn.Linear(self.net.classifier[-1].in_features, 2,
                                                self.net.classifier[-1].bias is not None)
        else:
            raise NotImplementedError()

    def forward(self, x):
        return self.net(x)


def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
