# UI 역할

from calculator.entity import Entity
from calculator.service import Service
from calculator.tf_service import TfService
class Controller:

    # 화면을 통해서 값이 들어 온 것으로 가정
    def execute(self, num1, num2, opcode):
        entity = Entity()
        entity.num1 = num1
        entity.num2 = num2
        entity.opcode = opcode

        service = Service(entity)
        if opcode == '+':
            result = service.add()
        if opcode == '-':
            result = service.subtract()
        if opcode == '*':
            result = service.multiply()
        if opcode == '/':
            result = service.divide()
        return result

        # 화면을 통해서 값이 들어 온 것으로 가정

    def execute_tf(self, num1, num2, opcode):
        entity = Entity()
        entity.num1 = num1
        entity.num2 = num2
        entity.opcode = opcode

        service = TfService(entity)
        if opcode == '+':
            result = service.add()
        if opcode == '-':
            result = service.subtract()
        if opcode == '*':
            result = service.multiply()
        if opcode == '/':
            result = service.divide()
        return result