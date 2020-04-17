# 실행하는 역할,  외부에서 보는 공간(실제 정보를 entity는 숨긴다.)
from calculator.controller import Controller
if __name__ == '__main__':
    def print_menu():
        print('0, Exit')
        print('1, Calculator')
        print('2, Tensor Calculator')
        return input('메뉴 선택 \n')


    while 1:
        menu = print_menu()
        if menu == '0':
            break
        if menu == '1':
            app = Controller()
            num1 = int(input('첫번째 수 \n'))
            opcode = input('연산자 \n')
            num2 = int(input('두번째 수 \n'))
            result = app.execute(num1,num2, opcode)
            print(f'{num1} {opcode} {num2} = {result}')

        if menu == '2':
            app = Controller()
            num1 = int(input('첫번째 수 \n'))
            opcode = input('연산자 \n')
            num2 = int(input('두번째 수 \n'))
            result = app.execute_tf(num1,num2, opcode)
            print(f'{num1} {opcode} {num2} = {result}')