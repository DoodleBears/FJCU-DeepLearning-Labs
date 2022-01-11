'''Information
This template is for Deep-learning class 2021.
Please read the assignment requirements in detail and fill in the blanks below.
Any part that fails to meet the requirements will deduct the points.
Please answer carefully.
'''

# Please import the required packages
import torch
import torch.nn as nn # neural networks
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as datasets # pytorch dataset

import numpy as np
import pandas as pd

# Define NeuralNetwork
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # image shape is 3 * 32 * 32, which
        # 3 is for RGB three-color channel, 32 * 32 is for image size

        # ------- convalution layer -------
        # please add at least one more layer of conv
        # ::: your code :::
        # NOTE: 卷积层 
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)

        # ::: end of code :::

        # ------- pooling layer and activation function -------
        # ::: your code :::
        self.pool = nn.MaxPool2d(kernel_size=, stride=)
        self.relu = nn.ReLU()
        # ::: end of code :::

        # ------- fully connected layers -------
        # ::: your code :::
        self.fc1 = nn.Linear( , 100)

        # The number of neurons in the last fully connected layer 
        # should be the same as number of classes

        # ::: end of code :::
        

    def forward(self, x):
        # first conv
        x = self.pool(self.relu(self.conv1(x))) # please count out the size for each layer
        # example: x = self.pool(self.relu(self.conv1(x))) # output size = 10x25x25
        # second conv
        # ::: your code :::
        # NOTE: Second conv 第二层卷积

        # ::: end of code :::

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        # fully connection layers
        # ::: your code :::
        # NOTE: FC 层 全连接

        # ::: end of code :::

        return x


def train():
    # Device configuration
    # ::: your code :::
    # NOTE: device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        
    # ::: end of code :::

    # set up basic parameters
    # ::: your code :::
    num_epochs = 10
    batch_size = 100
    learning_rate = 0.02
    # ::: end of code :::

    # step 0: import the data and set it as Pytorch dataset
    # Dataset: CIFAR10 dataset
    # NOTE: Dataset
    CIFAR10_train_data = datasets.CIFAR10('./data', train=True, download=True)
    CIFAR10_test_data = datasets.CIFAR10('./data', train=False, download=True)
    # NOTE: train_loader
    train_loader = DataLoader(dataset=CIFAR10_train_data, batch_size=batch_size)
    test_loader = DataLoader(dataset=CIFAR10_test_data, batch_size=batch_size)

    # step 1: set up models, criterion and optimizer
    model = ConvolutionalNeuralNetwork().to(device).train()
    # NOTE: criterion
    criterion = nn.BCELoss()
    # NOTE: optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    # step 2: start training
    n_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader): # run each step in i batch
            images, labels = images.to(device), labels.to(device)
            # ::: your code :::
            # init optimizer
            # 清零 gradient
            optimizer.zero_grad()

            y_predicted = model(images)
            

            # forward -> backward -> update
            # NOTE: Loss
            loss = criterion(y_predicted, labels)
            
            # backward
            loss.backward()


            # 更新 weight
            optimizer.step()

            

            
            # ::: end of code :::
        print(f'epoch {epoch+1}/{num_epochs}, loss = {loss.item():.4f}')

        # step 3: Validation loop
        # ::: your code :::
        # NOTE: Validation

        # ::: end of code :::
    print('Finished Training')

    # set model to Evaluation Mode
    # NOTE: set model to Evaluation Mode
    model = model
    

    # step 4: Testing loop
    # no grad here
    with torch.no_grad() :
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]
        # run through testing data
        
        # ::: your code :::
        for i, (images, labels) in enumerate(test_loader):

            # ::: end of code :::
            outputs = model(images)
            # max returns (value, index)
            _, predicted = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f'Accuracy of the network: {acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Accuracy of {i}: {acc} %')

    # save your model
    # ::: your code :::
    # TODO: save model

    # ::: end of code :::

if __name__ == '__main__':
    train()
