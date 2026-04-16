from io import BytesIO

import numpy as np
import torch

# Dataset


def get_float_tensor(data) -> torch.Tensor:
    return torch.tensor(data, dtype=torch.float32)


def get_dataset(is_test=False) -> tuple[torch.Tensor, torch.Tensor]:
    ds = np.lib.DataSource()
    fp = ds.open(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    )

    x = np.genfromtxt(
        BytesIO(fp.read().encode()), delimiter=",", usecols=range(2), max_rows=100
    )
    y = np.zeros(100)
    y[50:] = 1

    np.random.seed(1)
    idx = np.arange(y.shape[0])
    np.random.shuffle(idx)

    if is_test:
        X_test, y_test = x[idx[:25]], y[idx[:25]]
        return get_float_tensor(X_test), get_float_tensor(y_test)

    X_train, y_train = x[idx[25:]], y[idx[25:]]
    return get_float_tensor(X_train), get_float_tensor(y_train)


# Logistic Regression with Manual Gradients


def custom_where(cond, x_1, x_2):
    return (cond * x_1) + ((1 - cond) * x_2)


class LogisticRegression:
    def __init__(self, num_features: int):
        self.num_features = num_features
        self.weights = torch.zeros(num_features, 1, dtype=torch.float32)
        self.bias = torch.zeros(1, dtype=torch.float32)

    def forward(self, x: torch.Tensor):
        y = torch.add(torch.mm(x, self.weights), self.bias)
        probas = self._sigmoid(y)
        return probas

    def backward(self, probas: torch.Tensor, y: torch.Tensor):
        errors = y - probas.view(-1)
        return errors

    def predict_labels(self, x):
        probas = self.forward(x)
        labels = custom_where(probas >= 0.5, 1, 0)
        return labels

    def evaluate(self, x, y):
        labels = self.predict_labels(x).float()
        accuracy = torch.sum(labels.view(-1) == y) / y.size()[0]
        return accuracy

    def _sigmoid(self, z: torch.Tensor):
        return 1.0 / (1.0 + torch.exp(-z))

    def _logit_cost(self, y: torch.Tensor, proba):
        tmp1 = torch.mm(-y.view(1, -1), torch.log(proba))
        tmp2 = torch.mm((1 - y).view(1, -1), torch.log(1 - proba))
        return tmp1 - tmp2

    def train(self, x, y, num_epochs, learning_rate=0.01):
        for epoch in range(num_epochs):
            #### Compute outputs ####
            probas = self.forward(x)

            #### Compute gradients ####
            errors = self.backward(probas, y)
            neg_grad = torch.mm(x.transpose(0, 1), errors.view(-1, 1))

            #### Update weights ####
            self.weights += learning_rate * neg_grad
            self.bias += learning_rate * torch.sum(errors)

            #### Logging ####
            print("Epoch: %03d" % (epoch + 1), end="")
            print(" | Train ACC: %.3f" % self.evaluate(x, y), end="")
            print(" | Cost: %.3f" % self._logit_cost(y, self.forward(x)))


def logistic_regression_train():
    logr = LogisticRegression(num_features=2)

    X_train_tensor, y_train_tensor = get_dataset()
    logr.train(X_train_tensor, y_train_tensor, num_epochs=10, learning_rate=0.1)

    print("Model parameters:")
    print("  Weights:", logr.weights)
    print("  Bias:", logr.bias)


def logistic_regression_evaluate():
    logr = LogisticRegression(num_features=2)

    X_test_tensor, y_test_tensor = get_dataset(is_test=True)
    test_acc = logr.evaluate(X_test_tensor, y_test_tensor)
    print("Test set accuracy: %.2f%%" % (test_acc * 100))


# Logistic Regression using nn.Module


class LogisticRegressionModel(torch.nn.Module):
    def __init__(self, num_features: int):
        super().__init__()
        self.linear = torch.nn.Linear(num_features, 1)

    def forward(self, x: torch.Tensor):
        logits = self.linear(x)
        probas = torch.sigmoid(logits)
        return probas


def calc_accuracy(label_var, pred_probas):
    pred_labels = custom_where((pred_probas > 0.5).float(), 1, 0).view(-1)
    acc = torch.sum(pred_labels == label_var.view(-1)).float() / label_var.size(0)
    return acc


def logistic_model_train():
    model = LogisticRegressionModel(num_features=2)
    model.train()

    cost_fn = torch.nn.BCELoss(reduction="sum")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    num_epochs = 10

    X_train, y_train = get_dataset()

    X_train_tensor = X_train
    y_train_tensor = y_train.view(-1, 1)

    for epoch in range(num_epochs):
        #### Compute outputs ####
        out = model(X_train_tensor)

        #### Compute gradients ####
        cost = cost_fn(out, y_train_tensor)
        optimizer.zero_grad()
        cost.backward()

        #### Update weights ####
        optimizer.step()

        #### Logging ####
        pred_probas = model(X_train_tensor)
        acc = calc_accuracy(y_train_tensor, pred_probas)
        print("Epoch: %03d" % (epoch + 1), end="")
        print(" | Train ACC: %.3f" % acc, end="")
        print(" | Cost: %.3f" % cost_fn(pred_probas, y_train_tensor))

    print("\nModel parameters:")
    print(" | Weights: %s" % model.linear.weight)
    print(" | Bias: %s" % model.linear.bias)


def logistic_model_evaluate():
    model = LogisticRegressionModel(num_features=2)
    model.eval()

    X_test_tensor, y_test_tensor = get_dataset(is_test=True)

    pred_probas = model(X_test_tensor)
    test_acc = calc_accuracy(y_test_tensor, pred_probas)
    print("Test set accuracy: %.2f%%" % (test_acc * 100))


if __name__ == "__main__":
    pass
