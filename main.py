"""Train CIFAR10 with PyTorch."""

import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, transforms

from models import *
from utils import test_transforms, train_transforms

# parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")
# parser.add_argument("--lr", default=0.1, type=float, help="learning rate")
# parser.add_argument(
#     "--resume", "-r", action="store_true", help="resume from checkpoint"
# )
# parser.add_argument("--num_epochs", "-ne", action="store_true", help="number_of_epochs")
# args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch


class Cifar10SearchDataset(datasets.CIFAR10):

    def __init__(self, root="~/data", train=True, download=True, transform=None):

        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):

        image, label = self.data[index], self.targets[index]

        if self.transform is not None:

            transformed = self.transform(image=image)

            image = transformed["image"]

        return image, label


def TestTrainSplit(transform_train, transform_test, batch_size=512):
    trainset = Cifar10SearchDataset(
        "./data", train=True, download=True, transform=transform_train
    )
    testset = Cifar10SearchDataset(
        "./data", train=False, download=True, transform=transform_test
    )

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2
    )

    testloader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    return trainloader, testloader


# trainloader, testloader = TestTrainSplit(transform_train, transform_test)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)

# Model
print("==> Building model..")

net = ResNet18()

net = net.to(device)
if device == "cuda":
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True


criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)


from tqdm import tqdm


def train(model, device, train_loader, optimizer=optimizer, criterion=criterion):
    model.train()
    pbar = tqdm(train_loader)
    correct = 0
    processed = 0
    train_loss = 0
    for batch_idx, (data, target) in enumerate(pbar):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        y_pred = model(data)
        loss = criterion(y_pred, target)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        pred = y_pred.argmax(
            dim=1, keepdim=True
        )  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        pbar.set_description(
            desc=f"Loss={loss.item()} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
        )

    return (train_loss / len(train_loader)), (100 * correct / processed)


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(
                output, target, reduction="sum"
            ).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return test_loss, 100.0 * correct / len(test_loader.dataset)


def loop(
    num_epochs,
    trainloader,
    testloader,
    model=net,
    device=device,
    optimizer=optimizer,
    criterion=criterion,
    scheduler=scheduler,
):
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []
    for epoch in range(num_epochs):
        print("EPOCH:", epoch)
        loss, acc = train(
            model, device, trainloader, optimizer=optimizer, criterion=criterion
        )
        train_losses.append(loss)
        train_acc.append(acc)
        loss, acc = test(model, device, testloader)
        test_losses.append(loss)
        test_acc.append(acc)
        scheduler.step()
    return train_losses, test_losses, train_acc, test_acc


# loop(args.num_epochs)
