import torch

# Convolutional Neural Network


class CnnBasicModel(torch.nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()

        self.num_classes = num_classes

        self.features = torch.nn.Sequential(
            # 1@28x28 => 8@28x28
            torch.nn.Conv2d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,  # (1(28-1) - 28 + 3) / 2 = 1
            ),
            torch.nn.ReLU(),
            # 8@28x28 => 8@14x14
            torch.nn.MaxPool2d(
                kernel_size=(2, 2), stride=(2, 2), padding=0
            ),  # (2(14-1) - 28 + 2) = 0
            # 8@14x14 => 16@14x14
            torch.nn.Conv2d(
                in_channels=8,
                out_channels=16,
                kernel_size=(3, 3),
                stride=(1, 1),
                padding=1,  # (1(14-1) - 14 + 3) / 2 = 1
            ),
            torch.nn.ReLU(),
            # 16@14x14 => 16@7x7
            torch.nn.MaxPool2d(
                kernel_size=(2, 2),
                stride=(2, 2),
                padding=0,  # (2(7-1) - 14 + 2) = 0
            ),
        )

        self.output_layer = torch.nn.Linear(16 * 7 * 7, num_classes)

    def forward(self, x: torch.Tensor):
        x = self.features(x)  # d=[16,7,7]
        # x = torch.flatten(x, start_dim=0)  # d=[784]
        x = x.view(-1, 16 * 7 * 7)  # d=[1,784]
        x = self.output_layer(x)
        return x


def main_model_test():
    torch.manual_seed(123)

    x = torch.rand(28, 28)
    x = x.unsqueeze(dim=0)
    print("input shape:", x.shape)

    model = CnnBasicModel(num_classes=10)
    model.eval()

    with torch.no_grad():
        logits: torch.Tensor = model(x)
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    main_model_test()
