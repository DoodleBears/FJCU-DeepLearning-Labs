import pandas as pd
import numpy as np
import sys

def oneline_log(text):
    sys.stdout.write('\r')
    sys.stdout.write(text)

# LogisticRegression
class LogisticRegression():
    def __init__(self):
        super(LogisticRegression, self).__init__()

    def linear(self, x, w, b):

        return w * x.T + b

    def sigmoid(self, x):

        x = np.clip(x, -709.78, 709.78)

        return 1 / (1 + np.exp(-x))

    def forward(self, x, w, b):
        x = np.mat(x)
        w = np.mat(w)
        # print(f'x shape: {np.shape(x)}')
        # print(f'w shape: {np.shape(w)}')
        net_input = self.linear(x, w, b)
        # print(f'net_input: {net_input}')
        y_estimate = self.sigmoid(net_input)

        return y_estimate

model = LogisticRegression()

# BinaryCrossEntropy
class BinaryCrossEntropy():
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()
    
    def binary_cross_entropy(self, y_estimate, target):
        
        x = np.multiply(target , np.log(y_estimate + 1e-7)) + np.multiply((1 - target) , np.log(1 - y_estimate + 1e-7))

        return -(np.mean(x))

    def forward(self, y_estimate, target):

        return self.binary_cross_entropy(y_estimate, target)
criterion = BinaryCrossEntropy()


# generate random weight for each layer
# number of weight = input * neuron of next layer

# return value is 2D array of weight w_ji 
# w_ji means for the j neuron, the weight of input i
random_scalar = 100
def generate_layer_weight(seed, neuron, input):
    np.random.seed(seed) # set seed for weight random
    # w_ji 其中 j 对应 neuron, i 对应 input，所以 reshape 也同样按照如此进行
    weight = np.random.randn(neuron,input) / random_scalar
    # weight = np.zeros((neuron,input))
    return weight


# generate random bias for each layer
# number of bias = neuron of layer

# return value is vector of bias
def generate_layer_bias(seed, neuron):
    np.random.seed(seed) # set seed for bias random
    bias = np.random.randn(neuron, 1) / random_scalar
    # bias = np.zeros((neuron,1))
    return bias

def one_hot(value):
    if value == 9:
        return np.array([0,0,0,1])
    elif value == 8:
        return np.array([0,0,1,0])
    elif value == 3:
        return np.array([0,1,0,0])
    elif value == 0:
        return np.array([1,0,0,0])


pd_train_origin = pd.read_csv('data/lab3_train.csv')

# 保留 label 为 0、3、8、9 的 data
# 用 drop() 的方法，参见: https://www.cnblogs.com/everfight/p/pandas_condition_remove.html
pd_train_origin = pd_train_origin[(pd_train_origin.label == 0) 
                                | (pd_train_origin.label == 3) 
                                | (pd_train_origin.label == 8) 
                                | (pd_train_origin.label == 9) ]


pd_train = pd_train_origin.sample(frac=0.8, random_state=2)
pd_validate = pd_train_origin.drop(index=pd_train.index)

pd_validate.info()

pd_train.info()

# 处理数据，分离 feature 和 target
# 标准化数据（除以255）
train_feature = pd_train.drop(['label'], axis=1)
train_feature = train_feature / 255
train_target = pd.DataFrame(pd_train.label)
train_feature = np.array(train_feature)
train_target = np.array(train_target)

validate_feature = pd_validate.drop(['label'], axis=1)
validate_feature = validate_feature / 255
validate_target = pd.DataFrame(pd_validate.label)
validate_feature = np.array(validate_feature)
validate_target = np.array(validate_target)



# generate neuron for hidden layers
layer_neuron = [108, 36, 4]
all_layer_neuron = [784]
all_layer_neuron.extend(layer_neuron)
print(all_layer_neuron)

w = {} # weight of layers
b = {} # bias of layers
for i, neuron in enumerate(layer_neuron):
    input_size = 784 if i == 0 else layer_neuron[i-1]
    w[i+1] = generate_layer_weight(seed=2, neuron=neuron, input=input_size)
    print(f'layer {i}, weight shape: {np.shape(w[i+1])}')
    # print(w[i+1])
    b[i+1] = generate_layer_bias(seed=2, neuron=neuron)
    print(f'layer {i}, bias shape: {np.shape(b[i+1])}')


# -------------- hyper parameter -------------- 
epoch_num = 1000
learning_rate = 0.02
overfit_threshold = 0.01 # 如果 acc 比 max_validate_acc_count 小 overfit_threshold 的话
batch_size = 10
loss_threshold = 5e-5 # 设这么小大概是因为测过，train到这么小，validate_acc 还在变大


# 记录最佳情况
best_w = {} # best weight of layers
best_b = {} # best bias of layers
best_epoch = 0 # best epoch
best_train_loss = 0
best_validate_loss = 0

validate_size = len(validate_target)
train_size = len(train_target)
max_train_acc_count = 0
max_validate_acc_count = 0

is_overfit = False


# -------------- Train -------------- 
for epoch in range(epoch_num):
    train_loss_sum = 0
    # oneline_log(f'epoch {epoch + 1}')
    train_acc_count = 0
    oneline_log(f'epoch: {epoch+1}')
    for batch in range(train_size // batch_size):
        # 第 i 笔 data 的 feature
        a = {} # output of layers
        error = {} # error of layers
        a[0] = []
        for index in range(batch * batch_size, (batch+1) * batch_size):
            a[0].append(train_feature[index])
            
        a[0] = np.mat(a[0])
        # print(np.shape(a[0]))

        # Forward
        for layer, neuron in enumerate(layer_neuron):
            output = model.forward(a[layer], w[layer+1], b[layer+1])  
            a[layer+1] = output.T

        estimate_target = a[len(layer_neuron)]


        # Backward
        # 取得该次 batch 的 target
        batch_target = train_target[batch * batch_size : (batch+1) * batch_size]
        
        batch_target_onehot = []
        
        # 计算正确笔数
        for i in range(batch_size):
            # 将该次 batch 的 target 转为 onehot
            batch_target_one_hot = one_hot(batch_target[i][0])
            batch_target_onehot.append( batch_target_one_hot )
            if batch_target_one_hot[(np.argmax(estimate_target[i]))] == 1:
                train_acc_count += 1


        batch_target_onehot = np.mat(batch_target_onehot)

        # 计算 loss
        train_loss_sum += criterion.forward(estimate_target, batch_target_onehot)
        if epoch == 1:
            print(f'\tbatch {batch}: train_acc={train_acc_count / train_size}, train_loss={train_loss_sum/batch}')

        # Gradient
        error[len(layer_neuron)] = np.mat(estimate_target - batch_target_onehot).T

        for layer in range(len(layer_neuron) - 1, -1, -1):
            # error (4,100) (36,100) (784, 100)
            # weight (4,36) (784,36)
            # left (36,100) (784,100)
            left = np.mat(w[layer+1]).T * error[layer+1]

            right = np.multiply( a[layer].T, 1-a[layer].T)
            # print(f'layer: {layer}, error_shape: {error[layer+1].shape}, left_shape: {np.shape(left)}, right_shape: {np.shape(right)}')
            
            error[layer] = np.multiply(left , right)

        # Update parameter
        for layer in range(1, len(layer_neuron)+1):
            # print(f'layer: {layer}, error_shape: {error[layer].shape}')

            dw = np.dot(error[layer] , np.mat(a[layer-1]))
            w[layer] -= learning_rate * dw
            b[layer] -= learning_rate * error[layer].mean(axis=1)

    # 输出 validate 情况
    if (epoch+1) % 10 == 0:
        validate_acc_count = 0
        
        a = {} # output of layers
        error = {} # error of layers
        a[0] = validate_feature
            
        a[0] = np.mat(a[0])

        # Forward
        for layer, neuron in enumerate(layer_neuron):
            output = model.forward(a[layer], w[layer+1], b[layer+1])  
            a[layer+1] = output.T

        estimate_target = a[len(layer_neuron)]

        # Backward
        # 取得该次 batch 的 target
        batch_target = validate_target
        
        batch_target_onehot = []
        
        # 计算正确笔数
        for i in range(validate_size):
            # 将该次 batch 的 target 转为 onehot
            batch_target_one_hot = one_hot(batch_target[i][0])
            batch_target_onehot.append( batch_target_one_hot )
            if batch_target_one_hot[(np.argmax(estimate_target[i]))] == 1:
                validate_acc_count += 1


        batch_target_onehot = np.mat(batch_target_onehot)

        # 计算 loss
        validate_loss = criterion.forward(estimate_target, batch_target_onehot)

        train_loss = train_loss_sum / (train_size / batch_size)
        if max_train_acc_count < train_acc_count:
            max_train_acc_count = train_acc_count
            best_train_loss = train_loss
        if max_validate_acc_count < validate_acc_count:
            max_validate_acc_count = validate_acc_count
            best_validate_loss = validate_loss
            best_b = b
            best_w = w
            best_epoch = epoch

        oneline_log('')
        print(f'epoch: {epoch+1}: max_validate_acc = {max_validate_acc_count/validate_size}, train_acc={np.round(train_acc_count / train_size,4)}({train_acc_count}/{train_size}), validate_acc={np.round(validate_acc_count / validate_size,4)}({validate_acc_count}/{validate_size}), train_loss={train_loss}, validate_loss={validate_loss}')
        
        # 判断是否低于 threshold
        if train_loss_sum / (train_size / batch_size) < loss_threshold:
            print(f'stop because loss_threshold < {loss_threshold}')
            break
        # 判断是否 overfitting
        if max_validate_acc_count/validate_size - validate_acc_count/validate_size > overfit_threshold:
            is_overfit = True
            print('========== stop because overfitting ==========')
            break

print(f'best result at epoch {best_epoch + 1}: learning_rate={learning_rate}, neuron={all_layer_neuron}, validate_acc={max_validate_acc_count/validate_size}, validate_loss={best_validate_loss}, train_acc={max_train_acc_count/train_size}, train_loss={best_train_loss}')




# 针对 test 档案生成 ans
pd_test_origin = pd.read_csv('data/lab3_test.csv')
pd_test_origin = pd_test_origin / 255
pd_test_origin = np.array(pd_test_origin)


ans = []
for i, feature_data in enumerate(pd_test_origin):
    a = {} # output of layers
    error = {} # error of layers
    a[0] = feature_data

    # Forward
    for layer, neuron in enumerate(layer_neuron):
        output = model.forward(a[layer], best_w[layer+1], best_b[layer+1])            
        a[layer+1] = np.array(output.reshape(1,-1))[0]

    arr = a[len(layer_neuron)]

    for i, data in enumerate(arr):
        if data == np.max(arr):
            if i == 0:
                ans.append(0)
            elif i == 1:
                ans.append(3)
            elif i == 2:
                ans.append(8)
            elif i == 3:
                ans.append(9)      


test_ans = pd.DataFrame(ans)
test_ans = test_ans.rename({0:'ans'},axis=1)
test_ans.to_csv('test_ans.csv', index=None)

