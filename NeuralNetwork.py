
import numpy as np
import torch
import helper
import matplotlib.pyplot as plt

from torchvision import datasets, transforms

# Define a transform to normalize the data
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

# Download and load the training data
trainset = datasets.MNIST('MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()
print(type(images))
print(images.shape)
print(labels.shape)

plt.imshow(images[1].numpy().squeeze(), cmap='Greys_r');

def activation(x):
    return 1/(1+torch.exp(-x))

inputs = images.view(64,784)

weights_hidden = torch.randn(784,256)
weights_final=torch.randn(256,10)
bias_1=torch.randn(256)
bias_2=torch.randn(10)
hidden_output=activation(torch.mm(inputs,weights_hidden)+bias_1)
out = torch.mm(hidden_output,weights_final)+bias2 

