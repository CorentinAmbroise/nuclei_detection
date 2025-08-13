from torch import nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    """LeNet-5 architecture adapted for 256x256 RGB images"""

    def __init__(self, num_classes=2, dropout_rate=0.5):
        super(LeNet5, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 6, kernel_size=5, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)

        # Compute flattened size after convolutions for images 256x256
        # After conv1 + pool1: 256 -> 128
        # After conv2 + pool2: 124 -> 62
        # After conv3: 58
        # So: 120 * 58 * 58
        self.fc1 = nn.Linear(120 * 58 * 58, 84)
        self.fc2 = nn.Linear(84, num_classes)

        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Block 1
        x = F.relu(self.conv1(x))
        x = self.pool1(x)

        # Block 2
        x = F.relu(self.conv2(x))
        x = self.pool2(x)

        # Block 3
        x = F.relu(self.conv3(x))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
