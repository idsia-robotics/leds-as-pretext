import torch
from torchsummary import summary


class NN(torch.nn.Module):
    def __init__(self, in_channels=1, out_channels=2, classes=2):
        super(NN, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, 2, kernel_size=5,
                            stride=2, padding=2, bias=False),
            torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            torch.nn.BatchNorm2d(2),
            torch.nn.ReLU(),
            ConvBlock(2, 4, stride=2),
            ConvBlock(4, 6, stride=2),
            ConvBlock(6, 8, stride=2),
            ConvBlock(8, 16, stride=2)
        )

        self.pose_head = torch.nn.Sequential(
            torch.nn.Linear(400, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, out_channels),
            torch.nn.Sigmoid()
        )

        self.led_head = torch.nn.Sequential(
            torch.nn.Linear(400, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, classes),
            torch.nn.Softmax(dim=-1)
        )

    def forward(self, x):
        features = self.conv(x).flatten(1)
        return self.pose_head(features) * 320, self.led_head(features)


class ConvBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ConvBlock, self).__init__()

        self.layers = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
                            stride=stride, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3,
                            stride=1, padding=1, bias=False),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(),
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == '__main__':
    model = NN()
    summary(model, (1, 320, 320), device='cpu')
