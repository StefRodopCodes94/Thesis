import torch
from torchvision import datasets, transforms
import numpy as np

def pre(selected_labels):
    data_transform = transforms.Compose([transforms.ToTensor()])

    mnist_dataset = datasets.MNIST(root='./mnist_data', train=True, transform=data_transform, download=True)
    mnist_loader = torch.utils.data.DataLoader(mnist_dataset, batch_size=len(mnist_dataset), shuffle=False)

    # Get the data and labels from the dataset
    data, labels = next(iter(mnist_loader))

    selected_samples = {}
    num_samples_per_class = 2000

    for label in selected_labels:
        class_idx = torch.where(labels == label)[0]

        if len(class_idx) == 0:
            print(f"Class {label} not found in the dataset.")
            continue

        class_data = data[class_idx]

        # Calculate the mean and standard deviation for this class
        mean = class_data.mean(dim=0)
        std = class_data.std(dim=0)

        # Find samples within 1 standard deviation of the mean
        selected_idx = np.where(((class_data >= (mean - std)) & (class_data <= (mean + std))).all(dim=1))[0][:num_samples_per_class]

        selected_samples[label] = class_data[selected_idx]

    return selected_samples, selected_labels