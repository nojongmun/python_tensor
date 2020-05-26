# 경사 하강법(傾斜下降法, Gradient descent)은 1차 근삿값 발견용 최적화 알고리즘이다.
# 기본 개념은 함수의 기울기(경사)를 구하여 기울기가 낮은 쪽으로 계속 이동시켜서 극값에 이를 때까지 반복시키는 것이다.
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import matplotlib.pyplot as plt
class GradientDescent:
    @staticmethod
    def execute():
        X =  [1., 2., 3.] # 확률변수 3개
        Y =  [1., 2., 3.]
        m = len(X)
        # 경사 하강법 식
        W = tf.placeholder(tf.float32)
        hypothesis = tf.multiply(X, W)
        cost =  tf.reduce_mean(tf.pow(hypothesis - Y, 2)) / m  # 예측값과 정답의 차이값
        W_val = []
        cost_val = []
        with tf.Session() as sess:
            # 초기화
            init = tf.global_variables_initializer()
            sess.run(init)

            # 데이터 주입
            for i in range(-30, 50):
                W_val.append(i * 0.1)
                cost_val.append(sess.run(cost, {W:i * 0.1}))
            plt.plot(W_val, cost_val, 'ro')
            plt.ylabel('COST')
            plt.xlabel('W')
            plt.savefig('./data/result.svg')
            print('경사하강법 종료')
            return '종료'
if __name__ == '__main__':
    GradientDescent.execute()