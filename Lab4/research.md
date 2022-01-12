# 3. 实验内容
## a. 如何正确的进行资料读取

```python
import torchvision.transforms as transforms

transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.4750, 0.4750, 0.4750], std=[0.2008, 0.2008, 0.2008])
])
```
可以使用 torchvision 下的 transforms 在 DataLoader 处设置 transforms 参数进行处理

## b. Pytorch Function 有哪些参数可以调整以获得更好的准确率

- DataLoader 可以设置 shuffle 参数
- 在 load 数据的时候进行 Normalization 标准化
- optimizer 的 SGD 可以设置 momentum 权重衰减，weight_decay ，dampening 
- optimizer 可以使用 torch.optim.Adam 但效果似乎不一定更好

## c. 如何选择正确的硬体进行训练

通过判断设备是否支持 `GPU` 训练，如可以则可以设置 device 为 `cuda`，否则采 `CPU` 训练

```python
if torch.cuda.is_available():
    print('use cuda')
    device = torch.device("cuda")
else:
    print('use cpu')
    device = torch.device("cpu")
```

# 4. 实验讨论
## a. 模型参数的调参是如何去做的

主要在于调整 Conv layer 的参数和 pooling 的参数，结合不同的 `filter size`、`padding`、`input_channel`、`output_channel` 以及 不同的 `stride`

`learning_rate` 基本上一开始尝试较大的值，发现不会收敛后，调整为较小的值，收敛即可不用再过多的调整

## b. 不同学习率的表现
### learning_rate = 0.02 
不收敛，loss 持续在 2 左右徘徊

### learning_rate = 0.002 | 0.001
收敛，train_acc 能达到90%，validate_acc 能达到 65%, 在更深层更复杂的网络下可以达到 75%


## c. 不同 network 参数下的结果

- padding = 0, stride = 1
- batch_size = 5
- learning_rate = 0.001
- batch = 5

train_acc | val_acc  | test_acc  |     conv1     |  pool1  |     conv2     |  pool2  |     conv3     |  pool3  |  f1  |  f2  |  f3  |
----------|:--------:|----------:|------------------:|--------:|------------------:|--------:|------------------:|--------:|-----:|-----:|-----:|
76.471 %  | 65.340 % |         |  out=16,f=3  | f=2,s=2 |  out=16,f=5  | f=2,s=1 |  out=32,f=3  | f=2,s=2 |  120 |   84 |
74.013 %  | 66.840 % |         |  out=16,f=3  | f=2,s=2 |  out=16,f=5  | f=2,s=1 |  out=64,f=3  | f=2,s=2 |  120 |   84 |
81.109 %  | 67.940 % |  65.55 %  |  out=16,f=3  | f=2,s=2 |  out=32,f=5  | f=2,s=1 |  out=64,f=3  | f=2,s=2 |  120 |      |
71.247 %  | 66.700 % |  65.91 %  |  out=16,f=3  | f=2,s=2 |  out=32,f=5  | f=2,s=1 |  out=64,f=3  | f=2,s=1 |  120 |      |
88.733 %  | 75.060 % |  72.37 %  |  out=128,f=3  | f=2,s=2 |  out=192,f=5  | f=2,s=1 |  out=128,f=3  | f=2,s=1 |  120 |      |
88.733 %  | 75.060 % |  72.37 %  |  out=128,f=3  | f=2,s=2 |  out=192,f=5  | f=2,s=1 |  out=128,f=3  | f=2,s=1 |  120 |   84 |
92.531 %  | 74.640 % |  73.58 %  |  out=128,f=3  | f=2,s=2 |  out=192,f=5  | f=2,s=1 |  out=128,f=3  | f=2,s=1 |  120 |   84 |

- batch = 10

train_acc | val_acc  | test_acc  |     conv1     |  pool1  |     conv2     |  pool2  |     conv3     |  pool3  |     conv4     |  pool4  |  f1  |  f2  |  f3  |
----------|:--------:|----------:|------------------:|--------:|------------------:|--------:|------------------:|--------:|------------------:|--------:|-----:|-----:|-----:|
81.189 %  | 74.780 % |  74.74 % |  out=48,f=5  | f=2,s=2 |  out=128,f=3  | f=2,s=1 |  out=192,f=3  | f=2,s=1 |  out=128,f=3  | f=2,s=2 |  120 |      |
97.844 %  | 78.580 % |  77.40 % |  out=48,f=5  | f=2,s=2 |  out=128,f=3  | f=2,s=1 |  out=192,f=3  | f=2,s=1 |  out=128,f=3  | f=2,s=2 |  120 |  84  |

## d. 其他心得

在尝试各种 networks 的结构的时候，也有参考一些 state-of-the-art 的结构，或是经典的结构，如 `LeNet-5`、`VGG16`、`AlxeNet`等，和自己的实验结果类似，acc 的提升和 Neural Networks 的深度是呈正相关的，后期的网络往往越来越复杂，越来越深。