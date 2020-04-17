from contacts.controller import Controller
from contacts.service import Service
if __name__ == '__main__':

    def print_menu():
        print('0, Exit')
        print('1, 등록')
        print('2, 목록')
        print('3, 삭제')
        return input('메뉴 선택 \n')

    app = Controller()
    while 1:
        menu = print_menu()
        if menu == '0':
            break
        if menu == '1':
            app.register(input('이름 \n'),input('전화번호 \n'),input('이메일 \n'),input('주소 \n'))  # 피쳐의 갯수가 차원
        if menu == '2':
            print(app.list())
        if menu == '3':
            app.remove(input('이름 \n'))
        elif menu == '0':
            break