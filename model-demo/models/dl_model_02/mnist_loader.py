from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

BATCH_SIZE = 128
NUM_WORKERS = 4


def get_mnist_loader() -> tuple[DataLoader, DataLoader]:
    train_dataset = datasets.MNIST(
        root="./data", train=True, transform=transforms.ToTensor(), download=True
    )
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=True,
        shuffle=True,
    )

    test_dataset = datasets.MNIST(
        root="./data", train=False, transform=transforms.ToTensor()
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        drop_last=False,
        shuffle=False,
    )

    return train_loader, test_loader


def main_loader_test():
    train_loader, test_loader = get_mnist_loader()

    train_counter = Counter()
    for _, labels in train_loader:
        train_counter.update(labels.tolist())

    test_counter = Counter()
    for _, labels in test_loader:
        test_counter.update(labels.tolist())

    print("\nTraining label distribution:")
    sorted(train_counter.items())

    print("\nTest label distribution:")
    sorted(test_counter.items())


def main_loader_view():
    train_loader, _ = get_mnist_loader()

    for images, _ in train_loader:
        plt.figure(figsize=(8, 8))
        plt.axis("off")
        plt.title("Training images")
        plt.imshow(
            np.transpose(
                utils.make_grid(images[:64], padding=2, normalize=True),
                (1, 2, 0),
            )
        )
        plt.show()

        return


if __name__ == "__main__":
    pass
