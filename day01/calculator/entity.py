# 데이터 전송 역할 ,  (**I/O 역할) , DTO
# 모델을 제일 먼저 처리 할때 entity를 먼저 시작하자

class Entity:
    def __init__(self):
        # 인스턴스 변수 : _ => private 을 의미
        # entity 피쳐(특징)
        self._num1 = 0
        self._num2 = 0
        self._opcode = ''

    # getter 선언
    @property
    def num1(self) -> int:
        return self._num1

    # setter 선언
    @num1.setter
    def num1(self, num1):
        self._num1 = num1

        # getter 선언
    @property
    def num2(self) -> int:
        return self._num2

    # setter 선언
    @num2.setter
    def num2(self, num2):
        self._num2 = num2

    @property
    def opcode(self) -> int:
        return self._opcode

    # setter 선언
    @num2.setter
    def opcode(self, opcode):
        self._opcode = opcode