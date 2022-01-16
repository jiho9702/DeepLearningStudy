from audioop import cross
import numpy as np

def cross_entropy_error(y, t):
    delta = 1e-7
    #delta  ->  아주 작은 값을 더해줌으로써 np.log의 값이 0이 되지 않게끔 하고 값에 거의 변화가 없게 해준다.
    return -np.sum(t * np.log(y + delta))
    #x는 1일때 y는 0에 가까워지는 로그함수의 모습을 띄고있다.

t = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
y = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))

#0.510825457099338

y = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
print(cross_entropy_error(np.array(y), np.array(t)))
#2.302584092994546

#값이 클수록 정확도가 낮음을 보여준다.