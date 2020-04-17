from webcrawling.controller import Controller

if __name__ == '__main__':

    def print_menu():
        print('0, Exit')
        print('1, Bug Music')
        print('2, Naver Movie')
        print('3, 삭제')
        return input('메뉴 선택 \n')


    app = Controller()
    while 1:
        menu = print_menu()
        if menu == '0':
            break
        if menu == '1':
            app.bugs_music('https://music.bugs.co.kr/chart/track/realtime/total?chartdate=20200411&charthour=12')
        if menu == '2':
            app.naver_movie('https://movie.naver.com/movie/sdb/rank/rmovie.nhn')
        if menu == '3':
           pass
        elif menu == '0':
           break