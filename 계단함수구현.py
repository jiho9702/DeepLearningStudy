import numpy as np
import matplotlib.pylab as plt

# def step_function(x):
#     if x > 0:
#         return 1
#     else:
#         return 0

#넘파이 배열도 지원하도록 하는 함수
def step_function(x):
    y = x > 0
    return y.astype(np.int)

# x = np.array([-1.0, 1.0, 2.0])
# print(x)

# #Bool 타입으로 변경되어 0보다 큰 경우에만 True가 출력
# y = x > 0
# print(y)

# #Bool 을 int로 변경하여 True = 1, False = 0을 출력
# y = y.astype(np.int)
# print(y)

x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)
plt.show()
