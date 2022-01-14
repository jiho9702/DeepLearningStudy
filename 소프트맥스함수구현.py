# 소프트맥스 함수
# y = exp(특정 입력신호) / (sigma)(exp(입력신호 전체))
# exp(x) => 자연상수 e 의 x제곱

import numpy as np

a = np.array([0.3, 2.9, 4.0])
exp_a = np.exp(a)
print(exp_a)
#[ 1.34985881 18.17414537 54.59815003]

sum_exp_a = np.sum(exp_a)
print(sum_exp_a)
#74.1221542101633

y = exp_a / sum_exp_a
print(y)
#[0.01821127 0.24519181 0.73659691]

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y