# avgTemp,minTemp,maxTemp,rainFall,avgPrice  => 4개의 변수를 이용해서  1개의 결과 보기
# 보는 것(미완성) => 모델 , 저장(완성) => 머신

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from pandas.io.parsers import read_csv
import numpy as np
class Cabbage:
    def create_model(self):
        data = read_csv('data/price_data.csv', sep=',')
        xy = np.array(data, dtype=np.float32)
        x_data = xy[:, 1:-1]  # 입력데이터 첫 줄을 제외시키자
        y_data = xy[:, [-1]]  # 뒤에서 한칸 (원하는 정답)
        X = tf.placeholder(tf.float32, shape=[None, 4])  # 입력값 4개
        Y = tf.placeholder(tf.float32, shape=[None, 1])  # 출력값 1개
        W = tf.Variable(tf.random_normal([4, 1]), name='weight')  # 가중치
        b = tf.Variable(tf.random_normal([1]), name='bias')  # 누적값
        hypothesis = tf.matmul(X, W) + b      # 가설 : tf.matmul(X, W) 경사하강법,  + b (누적값)를 주지 않으면 무한 반복을 한다.
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for step in range(100000):
                cost_, hypo_, _ = sess.run([cost, hypothesis, train],   # 처음에 들어가는 값
                                           {X: x_data, Y: y_data})
                if step % 1000 == 0:
                    print("#", step, " 손실비용", cost_)
                    print("- 배추가격 : ", hypo_[0])
            saver = tf.train.Saver()
            save_path = './saved_model/model.ckpt'
            saver.save(sess, save_path)
'''            
if __name__ == '__main__':
    model = Cabbage()
    model.create_model()
'''