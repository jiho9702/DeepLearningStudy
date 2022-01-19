#f(x0, x1) = x0^2 + x1^2
#x0에 대한 미분인가 x1에 대한 미분인가
#변수가 여럿인 함수에 대한 미분을 편미분이라고 한다.
#동시에 구현:
import numpy as np

def function_2(x):
    return x[0]**2 + x[1]**2

def numerical_gradiant(f,x):
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

    return grad

print(numerical_gradiant(function_2, np.array([3.0, 4.0])))
print(numerical_gradiant(function_2, np.array([0.0, 2.0])))
print(numerical_gradiant(function_2, np.array([3.0, 0.0])))

#각 점의 기울기를 구할 수 있다.
#이때의 기울기는 화살표를 가진 벡터로 그려진다. 기울기 화살표는 함수의 "가장낮은장소(최솟값)"을 가리키는 것 같다
#화살표들은 한 점을 향하고 있는 모습니다.
#가장 낮은곳에서 멀리 떨어져있는 곳일수록 화살표의 크기가 커진다.
#정확하게 말하자면 각각의 지점에서 기울기가 낮아지는 방향을 가리킨다.
#즉 기울기가 가리키는 쪽은 각 장소에서 함수의 출력 값을 가장 크게 줄이는 방향이다.(쯍요!!!)