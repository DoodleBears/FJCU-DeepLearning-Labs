# a. 模型参数的调参是如何去做的

主要在调整 Conv layer 的参数和 pooling 的参数，结合不同的 filter size 和 output_channel

# b. 不同学习率的表现
## learning_rate = 0.02 
不收敛，loss 持续在 2 左右徘徊

## learning_rate = 0.002 
收敛，train_acc 能达到90%，validate_acc 能达到 65%


# c. 
train_acc | val_acc  | conv1 |  pool1  | conv2 |  pool2  | conv3 |  pool3  | conv4 |  pool4  |
----------|:--------:|------:|--------:|------:|--------:|------:|--------:|------:|--------:|
76.471 %  | 65.340 % |  f=3  | f=2,s=2 |  f=5  | f=2,s=1 |  f=3  | f=2,s=2 |       |         |
76.471 %  | 65.340 % |  f=3  | f=2,s=2 |  f=5  | f=2,s=1 |  f=3  | f=2,s=2 |       |         |
