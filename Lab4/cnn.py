'''Information
This template is for Deep-learning class 2021.
Please read the assignment requirements in detail and fill in the blanks below.
Any part that fails to meet the requirements will deduct the points.
Please answer carefully.
'''

# Please import the required packages
import torch
import torch.nn as nn # neural networks
from torch.utils.data import DataLoader, random_split
import torchvision.datasets as datasets # pytorch dataset
import torchvision.transforms as transforms
torch.manual_seed(233)

import numpy as np
import pandas as pd

from datetime import datetime

# Define NeuralNetwork
class ConvolutionalNeuralNetwork(nn.Module):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # image shape is 3 * 32 * 32, which
        # 3 is for RGB three-color channel, 32 * 32 is for image size

        # ------- convalution layer -------
        # please add at least one more layer of conv
        # ::: your code :::
        # NOTE: 卷积层 需要改成和书上不同的参数
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=48, kernel_size=5) # 3x32x32 -> 48x28x28
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=128, kernel_size=3, padding=1) # 48x14x14 -> 128x14x14
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, padding=1) # 128x13x13 -> 192x13x13
        self.conv4 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=3) # 192x12x12 -> 128x10x10

        # ::: end of code :::

        # ------- pooling layer and activation function -------
        # ::: your code :::
        # NOTE: pooling layer 池化层
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.relu = nn.ReLU()
        # ::: end of code :::

        # ------- fully connected layers -------
        # ::: your code :::
        # NOTE: FC Fully Connected 全连接层
        self.fc1 = nn.Linear(128*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        # The number of neurons in the last fully connected layer 
        # should be the same as number of classes
        self.fc3 = nn.Linear(84, 10)

        # ::: end of code :::
        

    def forward(self, x):
        # first conv
        x = self.pool1(self.relu(self.conv1(x))) # output size = 48x15x15
        # second conv
        # ::: your code :::
        # NOTE: 其他卷积层
        x = self.pool2(self.relu(self.conv2(x))) # output size = 128x14x14
        x = self.pool3(self.relu(self.conv3(x))) # output size = 192x7x7
        x = self.pool4(self.relu(self.conv4(x))) # output size = 128x4x4

        # ::: end of code :::

        # flatten all dimensions except batch
        x = torch.flatten(x, 1)
        # fully connection layers
        # ::: your code :::
        # NOTE: FC 层 全连接
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)

        # ::: end of code :::

        return x


   

def train():
    # Device configuration
    # ::: your code :::
    # NOTE: device
    if torch.cuda.is_available():
        print('use cuda')
        device = torch.device("cuda")
    else:
        print('use cpu')
        device = torch.device("cpu")
        
    # ::: end of code :::

    # set up basic parameters
    # ::: your code :::
    num_epochs = 100
    batch_size = 10
    learning_rate = 0.001
    # ::: end of code :::

    # step 0: import the data and set it as Pytorch dataset
    # Dataset: CIFAR10 dataset
    # NOTE: Dataset
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[0.2008, 0.2008, 0.2008])
    ])
    CIFAR10_train_data = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    CIFAR10_test_data = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    validate_size = 5000
    train_size = len(CIFAR10_train_data) - validate_size
    train_ds, val_ds = random_split(CIFAR10_train_data, [train_size, validate_size], seed=100)
    # NOTE: train_loader
    
    train_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    validate_loader = DataLoader(dataset=val_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=CIFAR10_test_data, batch_size=batch_size, shuffle=False)
    # step 1: set up models, criterion and optimizer
    model = ConvolutionalNeuralNetwork()
    print(model)
    model.to(device).train()
    
    # NOTE: criterion
    # criterion = nn.BCELoss()
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数

    # NOTE: optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    # step 2: start training
    n_total_steps = len(train_loader)
    best_train_acc = 0.0
    best_validate_acc = 0.0
    best_epoch = 0
    last_loss = 0.0
    loss_increase_time = 0 # for early-stop
    loss_increase_threshold = 2 # for early-stop
    early_stop = False # for early-stop
    for epoch in range(num_epochs):
        n_correct = 0
        n_samples = 0
        for i, (images, labels) in enumerate(train_loader): # run each step in i batch
            images, labels = images.to(device), labels.to(device)
            # ::: your code :::
            # init optimizer
            # 清零 gradient
            optimizer.zero_grad()
            # NOTE: Forward
            y_predicted = model(images)
            # calculate correct number
            _, predicted = torch.max(y_predicted, 1)
            n_samples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            # NOTE: Loss
            loss = criterion(y_predicted, labels)
            # NOTE: Backward
            loss.backward()
            # 更新 weight
            optimizer.step()
            # ::: end of code :::
        acc = 100.0 * n_correct / n_samples
        if acc > best_train_acc:
                best_train_acc = acc
        print(f'epoch {epoch+1}/{num_epochs}, train acc = {acc:.3f} %, loss = {loss.item():.4f}')
        # step 3: Validation loop
        # ::: your code :::
        # NOTE: Validation
        # NOTE: checkpoint save model 根据 epoch 保存 model
        date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")
        FILE = f'checkpoint/checkpoint_{date_time}_epoch_{epoch}.pt'
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'acc': acc,
            }, FILE)
        if (epoch+1) % 2 == 0:
            

            with torch.no_grad() :
                n_correct = 0
                n_samples = 0
                
                for i, (images, labels) in enumerate(validate_loader):
                    images, labels = images.to(device), labels.to(device)
                    outputs = model(images)
                    # max returns (value, index)
                    # 计算正确个数
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    n_samples += labels.size(0)
                    n_correct += (predicted == labels).sum().item()

                acc = 100.0 * n_correct / n_samples

                if acc > best_validate_acc:
                    best_validate_acc = acc
                    best_epoch = epoch + 1
                
                print(f'| Validation acc: {acc:.3f} %, loss = {loss.item():.4f}')
                if last_loss < loss.item() and last_loss != 0.0:
                    loss_increase_time += 1
                    if loss_increase_time >= loss_increase_threshold:
                        early_stop = True
                        break
                else:
                    loss_increase_time = 0

                last_loss = loss.item()
        

        # ::: end of code :::
    print('Finished Training')
    print(model)
    
    # set model to Evaluation Mode
    # NOTE: set model to Evaluation Mode
    model = model.eval()
    print(f'| Epoch: {num_epochs}')
    print(f'| Learning rate: {learning_rate}')
    print(f'| best epoch: {best_epoch}')
    print(f'| best train acc: {best_train_acc}')
    print(f'| best validate acc: {best_validate_acc}')
    print(f'| Early-stop: {early_stop}')
    

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
            images, labels = images.to(device), labels.to(device)
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
        print(f'Test acc: {acc} %')

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f'Test acc of {i}: {acc} %')

    # save your model
    # ::: your code :::
    # NOTE: 训练结束保存模型
    now = datetime.now()
    date_time = now.strftime("%m_%d_%Y_%H_%M_%S")
    FILE = f'model/model_{date_time}.pt'
    torch.save(model.state_dict(), FILE)
    # ::: end of code :::

if __name__ == '__main__':
    train()
