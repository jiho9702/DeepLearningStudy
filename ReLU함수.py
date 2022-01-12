import numpy as np

def relu(x):
    return np.maximum(0, x) #maximum함수
                            #두 값 중 큰 값을 출력해준다

print(relu(-5))