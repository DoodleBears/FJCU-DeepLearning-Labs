import pandas as pd
import numpy as np
import sys


def separate_label(data):
    data = pd.DataFrame(data)
    X = data.drop(['label'], axis=1)
    y = pd.DataFrame(data.label)
    return X, y


# Activation Function: Sigmoid
# 1/(1+e^(-n)) used to calculate the estimate value y
def sigmoid(n):
    return 1 / (1 + 1 / np.exp(n))

# Error Function: Cross-entropy loss
# used to calculate the loss of estimate
# a: estimation value of y
# y: true value of y
def calculate_cross_entropy(y, a):
    return -np.nan_to_num(np.multiply(y, np.log(a)) + np.multiply((1-y), np.log(1-a))).mean()

def calculate_mse(y, a):
    return np.square(np.subtract(y, a)).mean()

def calculate_rmse(y, a):
    return np.sqrt(calculate_mse(y, a))

def calculate_mae(y, a):
    return np.absolute(np.subtract(y, a)).mean()

# generate 784 weights in this case with specific seed
def generate_weight(seed):
    np.random.seed(seed) # set seed for weights random
    weights = np.random.randn(784,1) # in this case, we always need 784 random weights
    return np.matrix(weights)

# generate 1 bias in this case with specific seed
def generate_bias(seed):
    np.random.seed(seed) # set seed for bias random
    return np.random.rand()

def oneline_log(text):
    sys.stdout.write('\r')
    sys.stdout.write(text)

def calculate_acc(real, estimate):
    true_count = 0
    for index in range(len(real)):
        if real[index] == estimate[index]:
            true_count +=1
    acc = np.round(true_count / len(real) * 100.0,decimals=4)
    return acc

def validate_logistic_regression(data, weight, bias):
    validate_X, validate_y = separate_label(data)
    # print('|  VALIDATION DATA LENGTH: {}'.format(len(data)))
   
    
    # print('\n======================== START VALIDATING ========================\n')
    validate_X = validate_X / 255
    validate_X = np.array(validate_X)
    x = np.mat(validate_X).T # matrix construct by all pictures, each pic contain the 784 pixels as a vector
    validate_y = validate_y['label'].map({5:1, 2:0}) # map 5 to 1 and 2 to 0 for binary classification
    y = np.mat(validate_y)

    
    n = weight.T * x + bias
    estimate = sigmoid(n)
    estimate = np.round(estimate).astype(int)
    y = np.array(y)[0]
    estimate = np.array(estimate)[0]

    # print('y: {}'.format(y))
    # print('estimate: {}'.format(estimate))
    return calculate_acc(y, estimate=estimate)
    
# generate 1 bias in this case with specific seed
# data: data with label
# validate_data: validate_data with label
# epoch: max epoch (if reach, train process stop)
# stop error: if the error value smaller then stop error, train process stop
# error function: string which could be 'mse', 'mae', 'rmse'
# learning rate: 
def train_logistic_regression(data, validate_data , epoch, stop_error, early_stop_threshold, error_function, learning_rate):
    train_X, train_y = separate_label(data)
    
    weight = generate_weight(seed=3)
    bias = generate_bias(seed=1)
    print('|  LIMIT EPOCH: {}\n|  STOP ERROR: {}\n|  ERROR FUNCTION: {}\n|  LEARNING RATE: {}'.format(epoch, stop_error, error_function, learning_rate))
    print('|  EARLY STOP THRESHOLD: {}'.format(early_stop_threshold))
    print('|  TRAIN DATA LENGTH: {}'.format(len(data)))
    print('|  PIXELS NUMBERS: {}'.format(len(train_X.columns)))
    
    print('\n\n======================== START TRAINING ========================\n')
    train_X = train_X / 255
    train_X = np.array(train_X)
  
    x = np.mat(train_X).T # matrix construct by all pictures, each pic contain the 784 pixels as a vector
    train_y = train_y['label'].map({5:1, 2:0}) # map 5 to 1 and 2 to 0 for binary classification
    y = np.mat(train_y)

    max_acc = 0.0      
    for current_epoch in range(epoch): # loop every row in data (read in every picture consist of 784 pixels)
        
        n = weight.T * x + bias
        a = sigmoid(n)
        if error_function == 'mse':
            error = calculate_mse(y, a)
        elif error_function  == 'mae':
            error = calculate_mae(y, a)
        elif error_function == 'cross_entropy':
            error = calculate_cross_entropy(y, a)
        else:
            error_function = 'rmse'
            error = calculate_rmse(y, a)
        if(error < stop_error):
            break

        dw = (x * (a - y).T ) / len(data)
        db = np.ones((1,len(data))) * (a - y).T / len(data)
        db = np.array(db)[0][0]
        weight = weight - learning_rate * dw
        bias = (bias - learning_rate * db)
        if(current_epoch % 100 == 0):
            acc = validate_logistic_regression(validate_data, weight=weight, bias=bias)
            train_acc = validate_logistic_regression(data, weight=weight, bias=bias)
            # print('|  CURRENT EPOCH: {}'.format(current_epoch + 1))
            # print('EPOCH {}, bias : {}'.format(current_epoch+1, bias))
            # print('EPOCH {}, weight 0: {}'.format(current_epoch+1, np.array(weight)[0][0]))
            # print('EPOCH {}, weight 100: {}'.format(current_epoch+1, np.array(weight)[100][0]))
            temp_str = 'EPOCH {}, {}_error: {}'.format(current_epoch+1, error_function, error)
            # oneline_log(temp_str)
            temp_str += '    |  validate_acc: {}%'.format(acc)
            # oneline_log(temp_str)
            temp_str += '    |  train_acc: {}%'.format(train_acc)
            oneline_log(temp_str)

        if(acc > max_acc):
            max_acc  = acc
            max_acc_epoch = current_epoch
            max_acc_weight = weight
            max_acc_bias = bias
        elif max_acc - acc >= early_stop_threshold:
            break
        if(train_acc >= 100.0):
            break

           
    print('\n\n======================== STOP TRAINING ========================\n')
    print('|  TRAIN DATA LENGTH: {}'.format(len(data)))
    print('|  {}_error: {}'.format(error_function, error))
    print('|  max_acc: {}%'.format(max_acc))
    print('|  max_acc_epoch: {}'.format(max_acc_epoch))
    print('|  LIMIT EPOCH: {}\n|  STOP ERROR: {}\n|  ERROR FUNCTION: {}\n|  LEARNING RATE: {}'.format(epoch, stop_error, error_function, learning_rate))
    if max_acc - acc >= early_stop_threshold:
        print('|  STOP REASON: early-stop because overfitting, exist best acc: {}%, at epoch: {}'.format(max_acc, max_acc_epoch))
    elif error < stop_error :
        print('|  STOP REASON: stop-error')
    elif train_acc >= 100.0:
        print('|  STOP REASON: train acc reach 100%')
    else:
        print('|  STOP REASON: reach max epoch')
    return max_acc_weight, max_acc_bias, weight, bias

def ans_generate(data, weight, bias):
    validate_X = data
    validate_X = validate_X / 255
    validate_X = np.array(validate_X)
    x = np.mat(validate_X).T # matrix construct by all pictures, each pic contain the 784 pixels as a vector

    
    n = weight.T * x + bias
    estimate = sigmoid(n)
    estimate = np.round(estimate).astype(int)
    estimate = np.array(estimate)[0]
    ans = pd.DataFrame(estimate)

    return ans

def main():
    pd_train_origin = pd.read_csv('data/train.csv')
    pd_train = pd_train_origin.sample(frac=0.8, random_state=2)
    pd_validate = pd_train_origin.drop(index=pd_train.index)

    max_acc_weight, max_acc_bias, weight, bias = train_logistic_regression(data=pd_train_origin, validate_data=pd_validate, epoch=25000, stop_error=0.005, early_stop_threshold=0.3,  error_function='cross_entropy', learning_rate=0.1)
    acc = validate_logistic_regression(pd_validate, weight=max_acc_weight, bias=max_acc_bias)

    print('|  bias: {}'.format(bias))
    print('|  weight: {}'.format(weight))
    print('正确率: {}%'.format(acc))


    pd_test_origin = pd.read_csv('data/test.csv')
    pd_test_origin


    ans = ans_generate(pd_test_origin, weight=max_acc_weight, bias=max_acc_bias)
    ans = pd.DataFrame(ans[0].map({1:5, 0:2})) # map 1 to 5 and 0 to 2 

    ans = ans.rename({0:'ans'},axis=1)


    ans.to_csv('test_ans.csv', index=None)


if __name__ == '__main__':
    main()  