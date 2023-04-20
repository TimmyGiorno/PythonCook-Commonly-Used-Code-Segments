import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()

        else "mps"
        if torch.backends.mps.is_available()

        else "cpu"
    )
    print(f"Using {device} device.\n")
    return device


def download_mnist():
    # Download training data from open datasets.
    training_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return training_data, test_data



class NeuralNetwork(nn.Module):

    def __init__(self):
        super().__init__()

        # Flatten Layer: torch.nn.Flatten(start_dim=1, end_dim=-1)
        # [64, 1, 28, 28] -> [64, 784]
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


if __name__ == "__main__":

    #Hyperparameters.
    BATCH_SIZE = 64
    DEVICE = get_device()


    # Crate datasets.
    training_data, test_data = download_mnist()
    train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

    for X, y in test_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}.")
        print(f"Shape of y: {y.shape} {y.dtype}.\n")
        break
        # Shape of X [N, C, H, W]: torch.Size([64, 1, 28, 28])
        # Shape of y: torch.Size([64]) torch.int64


    model = NeuralNetwork().to(DEVICE)



