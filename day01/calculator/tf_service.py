# 라이브러리가 없으면 오류  => 해결 터미널 창에   >  pip install --upgrade tensorflow
import tensorflow as tf

class TfService:
    def __init__(self, payload):
        self._num1 = payload.num1
        self._num2 = payload.num2

    @tf.function
    def add(self):
        return tf.add(self._num1, self._num2)

    @tf.function
    def subtract(self):
        return tf.subtract(self._num1, self._num2)

    @tf.function
    def multiply(self):
        return tf.multiply(self._num1, self._num2)

    @tf.function
    def divide(self):
        return tf.divide(self._num1, self._num2)

