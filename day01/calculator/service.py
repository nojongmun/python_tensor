# enity.py 와 service.py  => Model
# payload => entity의 정보를 주고 받는 통로
class Service:
    def __init__(self, payload):
        self._num1 = payload.num1
        self._num2 = payload.num2

    def add(self):
        return self._num1 + self._num2

    def subtract(self):
        return self._num1 - self._num2

    def multiply(self):
        return self._num1 * self._num2

    def divide(self):
        return self._num1 / self._num2
