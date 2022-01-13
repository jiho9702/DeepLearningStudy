import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def identify_function(x):
    return x

def init_network():     #가중치와 편향을 초기화 해주고 이를 network라는 딕셔너리에 저장한다.
    network = {}        #각 층에 필요한 매개변수(가중치와 편향)를 저장
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])

    return network

def forward(network, x):        #입력신호를 출력으로 변환하는 처리과정을 구현
                                #신호가 순 방향으로 전달 됨(순전파)을 알리기 위해 forward로 이름을 지음
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identify_function(a3)

    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)