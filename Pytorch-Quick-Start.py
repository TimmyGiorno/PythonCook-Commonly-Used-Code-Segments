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

    training_data = datasets.FashionMNIST(
        # root: the path where the train/test data is stored.
        root="data",
        # train: specifies training or test dataset.
        train=True,
        # downloads the data from the internet if it’s not available at root.
        download=True,
        # specify the feature and label transformations.
        transform=ToTensor(),
    )

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


def train(dataloader, model, loss_fn, optimizer):

    size = len(dataloader.dataset)
    model.train()

    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")




def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == "__main__":

    DEVICE = get_device()

    #Hyperparameters.
    BATCH_SIZE = 64
    LEARNING_RATE = 1e-3
    EPOCHS = 5

    '''
    Crate datasets.
    PyTorch provides two data primitives: torch.utils.data.DataLoader and torch.utils.data.Dataset 
    that allow you to use pre-loaded datasets as well as your own data. 
    Dataset stores the samples and their corresponding labels, 
    and DataLoader wraps an iterable around the Dataset to enable easy access to the samples.
    '''
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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)


    for t in range(EPOCHS):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model, loss_fn)
    print("Done!")

    torch.save(model.state_dict(), "models/quick_start.pth")
    print("Saved PyTorch Model State to model.pth")

    model = NeuralNetwork()
    model.load_state_dict(torch.load("models/quick_start.pth"))

    classes = [
        "T-shirt/top",
        "Trouser",
        "Pullover",
        "Dress",
        "Coat",
        "Sandal",
        "Shirt",
        "Sneaker",
        "Bag",
        "Ankle boot",
    ]

    model.eval()
    x, y = test_data[0][0], test_data[0][1]
    with torch.no_grad():
        pred = model(x)
        predicted, actual = classes[pred[0].argmax(0)], classes[y]
        print(f'Predicted: "{predicted}", Actual: "{actual}"')



