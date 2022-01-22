#신경망을 예로들어 실제로 기울기를 구하는 코드 구현
import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.functions import softmax, cross_entropy_error
from common.gradiant import numerical_gradiant

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)

        return loss

net = simpleNet()
print(net.W)

#[[ 1.0545981   1.53426733  0.53543655]
#[ 0.04205633 -0.37172569 -0.20508036]]


