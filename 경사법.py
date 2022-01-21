#기계학습에서의 학습단계에서는 항상 최적의 매개변수를 찾아내야한다.
#신경망 역시 최적의 매개변수(가중치와 편향)를 학습시에 찾아내야한다.
#최적 = 손실함수가 최솟값이 될때의 매개변수 값이다.
#기울기를 이용해 함수의 최솟값을 찾으려 하는 방법이 경사법이다.
#함수의 값을 낮추는 방안을 제시하는 지표가 기울기이다.
#하지만 기울기가 가리키는 곳이 항상 최솟값이진 않는다.
#기울어진 방향이 꼭 최솟값을 가리키는 것은 아니지만 그 방향으로 가야 함소의 최솟값을 구할수 있다.
#따라서 기울기의 정보를 가지고 나아갈 뱡향을 잡아야 한다.

#--------------------------------------경사법----------------------------------------------#

# 현 위치에서 기울어진 방향으로 일정 거리만큼 이동한다
# 그런다음 이동한 곳에서도 마찬가지로 기울기를 구하고 또 그 기울거진 방향으로 나아가기를 반복한다.
# 이렇게 해서 함수의 값을 점차 줄이는 것이 경사법이다.

#------------------------------------------------------------------------------------------#

#구현#

import numpy as np

def numerical_gradiant(f,x):  #함수의 기울기를 반환해준다.
    h = 1e-4
    grad = np.zeros_like(x)         #x와 형상이 같고 그 원소가 모두 0인 배열을 생성

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = tmp_val + h
        fxh1 = f(x)
        x[idx] = tmp_val - h
        fxh2 = f(x)

        grad[idx] = (fxh1 - fxh2) / (2*h)
        x[idx] = tmp_val

    return grad                     #gradiant

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x

    for i in range(step_num):
        grad = numerical_gradiant(f, x)
        x -= lr * grad

    return x

#f = 최적화 하려는 함수, init_x = 초기값, lr = learning rate 학습률, step_num = 경사법에 따른 반복 횟수
#이 함수를 잘 이용하면 함수의 극소값을 구할 수 있고 더 나아가 최솟값도 구할 수 있다.

def function_2(x):
    return x[0]**2 + x[1]**2

init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x = init_x, lr = 0.1, step_num = 100))

#[-6.11110793e-10  8.14814391e-10]
#초기값을 [-3.0, 4.0]으로 설정한 후 경사법을 이용하여 최솟값을 탐색한다. 그 결과로 나온 값은 [0, 0]에 가까운 값이다.

#학습률이 너무 크거나 작으면 좋은 결과를 얻을 수 없다.

#학습률이 너무 큰 경우
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x = init_x, lr = 10.0, step_num = 100))
#[-2.58983747e+13 -1.29524862e+12]

#학습률이 너무 작은 경우
init_x = np.array([-3.0, 4.0])
print(gradient_descent(function_2, init_x = init_x, lr = 1e-10, step_num = 100))
#[-2.99999994  3.99999992]
#학습률이 


