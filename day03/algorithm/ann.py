# https://parksrazor.tistory.com/84
import tensorflow as tf
import math
import numpy as np
import matplotlib.pyplot as plt
class Neuron:
    @staticmethod
    def new_random_uniform_number(dim):
        # 균일분포 : max min사이값 동일분포에서 수 추출
        # [1] 결과값 shape 행, 열 차원의 수  2 * 3
        return tf.random.uniform([dim], 0, 1)

    @staticmethod
    def new_random_normal_number(dim):
        return tf.random.normal([dim], 0, 1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + math.exp(-x))

    def new_nueron(self):
        x = 0
        y = 1
        w = tf.random.normal([1], 0, 1)  # 가중치
        b = tf.random.normal([1], 0, 1)  # 편차
        for i in range(1000):
            nueron = self.sigmoid(x * w + 1 * b)
            error = y - nueron
            w = w + x * 0.1 * error
            b = b + 1 * 0.1 * error

            if i % 100 == 99:
                print(i, error, nueron)
        return nueron

    def sigmoid_tanh_relu(self):
        x = np.arange(-5, 5, 0.01)
        sigmoid_x = [self.sigmoid(z) for z in x]
        tanh_x = [math.tanh(z) for z in x]
        relu = [0 if z < 0 else z for z in x]
        return {'x': x, 'sigmoid_x': sigmoid_x, 'tanh_x': tanh_x, 'relu': relu}

if __name__ == '__main__':
    nue = Neuron()
    neuron = nue.new_nueron()
    dic = nue.sigmoid_tanh_relu()
    # 문제는 0 답은 1
    x = dic['x']
    sigmoid_x = dic['sigmoid_x']
    tanh_x = dic['tanh_x']
    relu = dic['relu']
    plt.axhline(0, color='gray')
    plt.axvline(0, color='gray')
    plt.plot(x, sigmoid_x, 'b-', label='sigmoid')
    plt.plot(x, tanh_x, 'r--', label='tanh')
    plt.plot(x, relu, 'g.', label='relu')
    plt.legend()
    plt.show()