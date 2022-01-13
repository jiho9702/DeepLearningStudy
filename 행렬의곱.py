import numpy as np

A = np.array([[1,2], [3,4]])
print(A.shape)

B = np.array([[5,6], [7,8]])
print(B.shape)

print(np.dot(A, B))
#[[19 22]
# [43 50]]

C = np.array([[1,2,3], [4,5,6]])
D = np.array([[1,2], [3,4], [5,6]])

print(np.dot(C, D))

#[[22 28]
# [49 64]]

E = np.array([[1,2], [3,4]])
#print(np.dot(C, E))  곱셈을 할 수 가 없다.

#신경망에서의 행렬 곱
X = np.array([1,2])                 # X=입력노드 
W = np.array([[1,3,5], [2,4,6]])    # W=가중치
Y = np.dot(X, W)                    # Y=출력노드
print(Y)
#[ 5 11 17]
