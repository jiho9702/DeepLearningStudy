import numpy as np
A = np.array([1, 2, 3, 4])
print(A)

print(np.ndim(A))   #ndim => 배열의 차원 수

print(A.shape)      #배열의 형상, 인스턴스 변수인 shape으로 알수있다.

print(A.shape[0])   

B = np.array([[1,2], [3,4], [5,6]])
print(B)

print(np.ndim(B))

print(B.shape)