from torch import nn


def mlp(nn_sizes: list(), activation: list()) -> nn.Sequential():
    """
    feature_sizes: a list of the size of each layer of the mlp
    activation: a list of the activation functions for each layer of the mlp
    ------------------------------------------------------------------------
    Example:
    input size of 8, output size of 4, 2 hidden layers of size 16
    nn_size = [8, 16, 16, 4]

    all layers have ReLU activation
    activation = [nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU]
    """
    layers = []

    for i in range(len(nn_sizes) - 1):
        act = activation[i]
        layer_i = nn.Linear(nn_sizes[i], nn_sizes[i + 1])
        layers.append(layer_i)
        layers.append(act())

    mlp_model = nn.Sequential(*layers)

    return mlp_model


def main():
    nn_sizes = [8, 16, 16, 4]
    activation = [nn.ReLU, nn.ReLU, nn.ReLU, nn.ReLU]
    mlp_model = mlp(nn_sizes, activation)
    print(mlp_model)


if __name__ == "__main__":
    main()
