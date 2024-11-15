import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# Partie 0:
class MLPQuant(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPQuant, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x


def load_mnist():
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('data', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader


def model_train(model, train_loader, criterion, optimizer, epochs):
    for epoch in range(epochs):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            images = images.view(-1, 28 * 28)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if (i + 1) % 100 == 0:
                print(f'Epoch: {epoch + 1}, Batch: {i + 1}, Loss: {loss.item()}')
    print('Finished Training')


def model_test(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.view(-1, 28 * 28)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total}')
    return correct / total


def quantize_simple(model):
    for name, param in model.state_dict().items():
        param.copy_(torch.round(param).int())
    return model


if __name__ == "__main__":
    train_loader, test_loader = load_mnist()

    print("Normal model")
    model = MLPQuant(28 * 28, 128, 10)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    model_train(model, train_loader, criterion, optimizer, epochs=1)

    model_test(model, test_loader)

    print("\nQuantized model")
    model = quantize_simple(model)

    model_test(model, test_loader)

    # Cela ne fonctionne pas car la fonction de quantification n'est pas correcte.


# Partie 1:
def quantize(float_value, min_range, max_range, zero=0):
    b = 8  # number of bits for int8
    S = (max_range - min_range) / (2 ** b - 1)
    quantized = torch.round(float_value / S).to(torch.int8)
    return quantized


def quantize_model_unique_range(model):
    all_params = torch.cat([param.view(-1) for param in model.parameters()])
    min_val, max_val = all_params.min(), all_params.max()
    for name, param in model.state_dict().items():
        param.copy_(quantize(param, min_val, max_val))
    return model


def quantize_model_layer_range(model):
    for name, param in model.state_dict().items():
        min_val, max_val = param.min(), param.max()
        param.copy_(quantize(param, min_val, max_val))
    return model


if __name__ == "__main__":
    train_loader, test_loader = load_mnist()

    print("Normal model")
    model = MLPQuant(28 * 28, 128, 10)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    model_train(model, train_loader, criterion, optimizer, epochs=1)
    model_test(model, test_loader)

    print("\nQuantized model with unique range")
    model = quantize_model_unique_range(model)
    model_test(model, test_loader)

    print("\nQuantized model with layer-specific range")
    model = quantize_model_layer_range(model)
    model_test(model, test_loader)


class CNNQuant(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLPQuant, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x
