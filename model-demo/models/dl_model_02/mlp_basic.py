import torch

# Multilayer Perceptron


class MlpBasicModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_units: list[int], num_classes: int):
        super().__init__()

        # Initialize MLP layers
        all_layers = []
        for hidden_unit in hidden_units:
            layer = torch.nn.Linear(input_size, hidden_unit, bias=False)
            all_layers.append(layer)
            all_layers.append(torch.nn.ReLU())
            input_size = hidden_unit

        output_layer = torch.nn.Linear(
            in_features=hidden_units[-1], out_features=num_classes
        )

        all_layers.append(output_layer)
        self.layers = torch.nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor):
        x = torch.flatten(x, start_dim=1)  # to make it work for image inputs
        x = self.layers(x)
        return x


def model_test():
    torch.manual_seed(123)

    x = torch.rand(1, 28, 28)

    model = MlpBasicModel(input_size=28 * 28, hidden_units=[128, 256], num_classes=10)
    model.eval()

    with torch.no_grad():
        logits = model(x)
    print("logits shape:", logits.shape)


if __name__ == "__main__":
    model_test()
