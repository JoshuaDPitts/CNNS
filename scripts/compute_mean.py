from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch, numpy as np

dataset = CIFAR10(root="data", train=True, download=True,
                  transform=transforms.ToTensor())
loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

total = torch.zeros(3)
for imgs, _ in loader:
    total += imgs.sum([0, 2, 3])
mean = total / (len(dataset) * 32 * 32)
np.save("data/cifar10_mean.npy", mean.numpy())
print("Mean RGB:", mean)

