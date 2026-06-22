import torch
import torch.nn as nn

class EEGNet(nn.Module):
    """
    EEGNet model implementation in PyTorch.
    Based on various open-source implementations of Lawhern et al., 2018.
    
    Parameters:
        nb_classes (int): Number of classes to classify (e.g., 2 for Open/Close).
        Chans (int): Number of EEG channels (e.g., 20 for CGX).
        Samples (int): Number of time samples per epoch (e.g., 128 for 0.5s @ 250Hz).
        dropoutRate (float): Dropout rate.
        kernLength (int): Length of temporal convolution kernel.
        F1 (int): Number of temporal filters.
        D (int): Depth multiplier (number of spatial filters = F1 * D).
        F2 (int): Number of pointwise filters (F2 = F1 * D).
        norm_rate (float): CheckMaxNorm constraint rate (handled in training loop ideally, here just model def).
    """
    def __init__(self, nb_classes=2, Chans=26, Samples=256, 
                 dropoutRate=0.25, kernLength=64, F1=8, 
                 D=2, F2=16, norm_rate=0.25):
        super(EEGNet, self).__init__()
        
        self.F1 = F1
        self.D = D
        self.F2 = F2
        self.nb_classes = nb_classes
        self.Chans = Chans
        self.Samples = Samples

        # Block 1
        self.conv1 = nn.Conv2d(1, F1, (1, kernLength), padding=(0, kernLength // 2), bias=False)
        self.batchnorm1 = nn.BatchNorm2d(F1, False)
        
        # Depthwise Conv2D (Spatial Filter)
        self.conv2 = nn.Conv2d(F1, F1 * D, (Chans, 1), groups=F1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(F1 * D, False)
        self.activation = nn.ELU()
        self.avg_pool1 = nn.AvgPool2d((1, 4))
        self.dropout1 = nn.Dropout(dropoutRate)
        
        # Block 2
        # PyTorch doesn't have a direct 'SeparableConv2d'. We simulate it below.
        # We simulate it with Grouped Conv or implementing Depthwise + Pointwise.
        # The standard implementation often just uses a standard Conv2d or separate steps.
        # Here we follow a standard approximation:
        self.separable_conv = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False)
        )
        
        self.batchnorm3 = nn.BatchNorm2d(F2, False)
        self.avg_pool2 = nn.AvgPool2d((1, 8))
        self.dropout2 = nn.Dropout(dropoutRate)
        
        # Classification Block
        # Calculate input size for linear layer based on pooling
        # Samples -> /4 -> /8 = /32
        self.flatten_size = F2 * (Samples // 32)
        self.classifier = nn.Linear(self.flatten_size, nb_classes)

    def forward(self, x):
        # Input shape: (Batch, 1, Chans, Samples)
        
        # Block 1
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.activation(x)
        x = self.avg_pool1(x)
        x = self.dropout1(x)
        
        # Block 2
        x = self.separable_conv(x)
        x = self.batchnorm3(x)
        x = self.activation(x)
        x = self.avg_pool2(x)
        x = self.dropout2(x)
        
        # Classification
        x = x.view(-1, self.flatten_size)
        x = self.classifier(x)
        
        return x

if __name__ == "__main__":
    # Test valid input
    model = EEGNet(Chans=20, Samples=256)
    dummy_input = torch.randn(1, 1, 20, 256)
    output = model(dummy_input)
    print(f"Model Output Shape: {output.shape}")
