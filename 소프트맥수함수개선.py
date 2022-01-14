#softmax 함수의 구현은 잘 되지만 큰 값들이 계산되었을 경우 오버플로우가 발생한다
#컴퓨터가 표현할 수 있는 범위를 초과하여 생기는 현상을 오버플로우라고 한다.
#ex 4바이트를 출력할 수 있다고 가정 -> 8바이트가 출력되어야 한다 -> 4바이트의 손실 발생     8-4 = 4
#따라서, 입력신호 중 최대값을 사용하여 오버플로우를 막아준다.
#가장 큰 값을 행렬의 각 원소에 빼준다.

#구현

import numpy as np

a = np.array([1010, 1000, 990])
print(np.exp(a) / np.sum(np.exp(a)))
#[nan nan nan]

c = np.max(a)
print(a - c)
#[  0 -10 -20]

print(np.exp(a - c) / np.sum(np.exp(a - c)))
#[9.99954600e-01 4.53978686e-05 2.06106005e-09]

#nan으로 나타낼 수 없다고 한 것들을 표현 가능하다.

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

D = np.array([0.3, 2.9, 4.0])
y = softmax(D)
print(y)
#[0.01821127 0.24519181 0.73659691]
print(np.sum(y))
#1.0
# 다 더한 값이 1 나온다는 것은 각각을 확률로 봐도 된다.
# y[0]의 확률은 0.018%    y[1]의 확률은 0.245%    y[2]의 확률은 0.736%이다.
#2번째 원소의 확률이 가장 높으니 답은 2번째 클래스다 라고 할 수 있다.

