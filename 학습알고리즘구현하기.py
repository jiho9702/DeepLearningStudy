#-----------------------------------------------------------#
#   params = 신경망의 매개변수를 보관하는 딕셔너리 변수(인스턴스 변수)
#   params['W1'] = 1번째 층의 가중치, params['b1'] = 1번째 층의 편향
#   params['W2'] = 2번째 층의 가중치, params['b2'] = 2번째 층의 편향

#   grads = 기울기를 보관하는 딕셔너리 변수(numerical_gradient() 메서드의 반환 값)
#   grads['W1'] = 1번째 층의 가중치의 기울기, grads['b1'] = 1번째 층의 편향의 기울기
#   grads['W2'] = 2번째 층의 가중치의 기울기, grads['b2'] = 2번째 층의 편향의 기울기

#   __init__(self, input_size, hidden_size, output_size) = 초기화를 수행한다.
#   인수는 순서대로 입력층의 뉴런 수, 은닉층의 뉴런 수, 출력층의 뉴런 수

#   predict(self, x) = 예측(추론)을 수행한다. 인수 x는 이미지 데이터

#   loss(self, x, t) = 손실 함수의 값을 구한다. 인수 x는 이미지 데이터, t는 정답 레이블(아래 간의 세 메서드의 인수들도 마찬가지)

#   accuarcy(self, x, t) = 정확도를 구한다.

#   numerical_gradiant(self, x, t) = 가중치의 매개변수 기울기를 구한다.

#   gradiant(self, x, t) =  가중치의 매개변수 기울기를 구한다. numerical_gradiant()의 성능 개선판!
#-------------------------------------------------------------#
from audioop import cross
import sys, os
sys.path.append(os.pardir)
from common.functions import *
from common.gradiant import numerical_gradiant

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = 0.01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        t = np.argmax(t, axis = 1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradiant(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradiant(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradiant(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradiant(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradiant(loss_W, self.params['b2'])

        return grads


net = TwoLayerNet(input_size = 784, hidden_size= 100, output_size= 10)
print(net.params['W1'].shape)   #(784, 100)
print(net.params['b1'].shape)   #(100,)
print(net.params['W2'].shape)   #(100, 10)
print(net.params['b2'].shape)   #(10,)