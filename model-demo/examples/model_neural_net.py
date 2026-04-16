import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# auto gradient


def test_auto_gradient():
    # logistic regression classifier
    y = torch.tensor([1.0])
    x1 = torch.tensor([1.1])
    w1 = torch.tensor([2.2], requires_grad=True)
    b = torch.tensor([0.0], requires_grad=True)

    z = x1 * w1 + b
    a = torch.sigmoid(z)

    loss = F.binary_cross_entropy(a, y)
    print("loss:", loss)

    loss.backward()
    print("w grad:", w1.grad)
    print("b grad:", b.grad)


# multilayer neural network


class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs: int, num_outputs: int) -> None:
        super().__init__()

        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),
            # 2nd hidden layer
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),
            # output layer
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.layers(x)
        return logits


def test_neural_network():
    torch.manual_seed(123)

    model = NeuralNetwork(50, 3)
    print(model)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("\ntotal number of trainable model parameters:", num_params)

    print("\nlayer0 weight shape:", model.layers[0].weight.shape)
    print("layer0 weight:\n", model.layers[0].weight)


# data loader


class ToyDataset(Dataset):
    def __init__(self, X: torch.Tensor, y: torch.Tensor) -> None:
        self.features = X
        self.labels = y

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self) -> int:
        return self.labels.shape[0]


def test_toy_data_loader():
    torch.manual_seed(123)
    X_train = torch.tensor(
        [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1])
    print(f"X_train.shape={X_train.shape}, y_train.shape={y_train.shape}")

    train_ds = ToyDataset(X_train, y_train)
    print("dataset length:", len(train_ds))

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    print("\ndataloader length:", len(train_loader))

    for idx, (x, y) in enumerate(train_loader):
        print(f"batch {idx+1}:", x, y)


def get_train_dataloader() -> DataLoader:
    X_train = torch.tensor(
        [[-1.2, 3.1], [-0.9, 2.9], [-0.5, 2.6], [2.3, -1.1], [2.7, -1.5]]
    )
    y_train = torch.tensor([0, 0, 0, 1, 1])
    train_ds = ToyDataset(X_train, y_train)

    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        drop_last=True,
    )
    return train_loader


def get_test_dataloader() -> DataLoader:
    X_test = torch.tensor([[-0.8, 2.8], [2.6, -1.6]])
    y_test = torch.tensor([0, 1])
    test_ds = ToyDataset(X_test, y_test)

    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
        num_workers=0,
    )
    return test_loader


# training loop


def test_train_loop():
    torch.manual_seed(123)
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    train_loader = get_train_dataloader()

    model.train()
    num_epochs = 3

    for epoch in range(num_epochs):
        for batch_idx, (features, labels) in enumerate(train_loader):
            logits = model(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()  # compute gradients
            optimizer.step()  # update model weights

            print(
                f"epoch: {epoch+1:03d}/{num_epochs:03d}"
                f" | batch: {batch_idx+1:03d}/{len(train_loader):03d}"
                f" | train/val loss: {loss:.2f}"
            )

    fpath = "/tmp/test/model.pth"
    torch.save(model.state_dict(), fpath)
    print("\nmodel saved:", fpath)


# model evaluate


def test_evaluate_model():
    model = NeuralNetwork(num_inputs=2, num_outputs=2)

    fpath = "/tmp/test/model.pth"
    model.load_state_dict(torch.load(fpath, weights_only=True))
    print("model loaded:", fpath)

    model.eval()

    train_loader = get_train_dataloader()

    correct = 0
    total_examples = 0

    for batch_idx, (features, labels) in enumerate(train_loader):
        print(f"\npred batch: {batch_idx+1:03d}/{len(train_loader):03d}")
        with torch.no_grad():
            logits = model(features)
        print("logits:", logits)

        # torch.set_printoptions(sci_mode=False)
        probas = torch.softmax(logits, dim=1)
        print("probas:", probas)

        predictions = torch.argmax(probas, dim=1)
        print("pred:", predictions)

        correct += torch.sum(predictions == labels).item()
        total_examples += len(predictions)

    print("\nacc:", correct / total_examples)


if __name__ == "__main__":
    test_auto_gradient()

    # test_neural_network()
    # test_toy_data_loader()

    # test_train_loop()
    # test_evaluate_model()
