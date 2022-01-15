import sys, os
from matplotlib.cbook import flatten
sys.path.append(os.pardir)
from dataset.mnist import load_mnist
import numpy as np
from PIL import Image
import pickle

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize= True, flatten = True, one_hot_label= False)
    return x_test, t_test

def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y

x, t = get_data()
network = init_network()

batch_size = 100
accuracy_cnt = 0
network = init_network()

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))


#-----------------------------------------------------------#
#배치처리
#전체적으로 원소 784개로 구성된 1차원 배열(28 x 28)이 입력되어 마지막에는 원소가 10개인 1차원 배열이 출력되는 흐름이다.
#이미지가 한장인 경우 1 x 784이다.
#이미지가 여러장을 한꺼번에 입력하는 경우 => 이미지 100장을 predict로 넘겨준다 == 데이터를 받아들이는 x의 형상은 100 x 784가 된다
#이 경우 마지막까지의 연산을 거치면 100 x 10의 모습이 된다.
#이렇게 100개의 데이터를 하나로 묶은 형태를 배치(batch)라고 한다.  이미지가 지폐다발처럼 묶여있다.
x, _ = get_data()
network = init_network()
W1, W2, W3 = network['W1'], network['W2'], network['W3']
print(x.shape)              #(10000, 784)
print(x[0].shape)           #(784,)
print(W1.shape)             #(784, 50)
print(W2.shape)             #(50, 100)
print(W3.shape)             #(100, 10)

x = np.array([[0.1, 0.8, 0.1], [0.3, 0.1, 0.6], [0.2, 0.5, 0.3], [0.8, 0.1, 0.1]])      
y = np.argmax(x, axis = 1)
print(y)            #[1 2 1 0]

t = np.array([1, 2, 0, 0])
print(y == t)                   #실제 답으로 구성된 t배열과 y배열과의 비교를 하고  == 연산자를 사용해 넘파이 배열끼리 비교하여 True, False로
                                #구성된 bool배열을 반들고 True가 몇개인지 센다.
print(np.sum(y == t))