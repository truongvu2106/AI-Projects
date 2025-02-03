import torch
import torch.nn as nn


# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # YOUR CODE HERE
        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))

        self.backbone = nn.Sequential(
            # convolutional layer 1. It sees 3x224x224 image tensor
            # and produces 16 feature maps 224x224 (i.e., a tensor 32x224x224)
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 32x112x112
            # convolutional layer
            nn.Conv2d(32, 64, 3, padding=1),  # -> 64x112x112
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 64x56x56
            # convolutional layer
            nn.Conv2d(64, 128, 3, padding=1),  # -> 128x56x56
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 128x28x28
            # convolutional layer
            nn.Conv2d(128, 256, 3, padding=1),  # -> 256x28x28
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 256x14x14
            # convolutional layer
            nn.Conv2d(256, 512, 3, padding=1),  # -> 512x14x14
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # -> 512x7x7
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.BatchNorm1d(4096),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(4096, 1000),
            nn.BatchNorm1d(1000),
            nn.Dropout(p=dropout),
            nn.ReLU(),
            nn.Linear(1000, num_classes)
        )
        
        self.model = nn.Sequential(
            self.backbone,
            self.fc
        )


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # YOUR CODE HERE: process the input tensor through the
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)
        return self.model(x)


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])
    images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
