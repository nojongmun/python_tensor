from titanic.model import Titanic
from titanic.view import View
from titanic.controller import Controller

if __name__ == '__main__':
    def print_menu():
        print('0.  Exit')
        print('1.  Machine Learning')
        print('2.  View: Survived vs. Dead')
        print('3.  Test Accuracy')
        print('4.  Submit')
        return input('Choose One\n')

    while 1:
        menu = print_menu()
        print('Menu : %s ' % menu)
        if menu == '0':
            break
        if menu == '1':
            app = Controller()
            train = app.create_train()
            model = app.create_model(train)
        if menu == '2':
            view = View()
            temp = view.create_train()
            view.plot_survivled_dead(temp)
            # view.plot_sex(temp)
            # view.bar_chart(temp, 'Pclass')
        if menu == '3':
            app = Controller()
            train = app.create_train()
            model = app.create_model(train)
            dummy = app.create_dummy(train)
            app.test_all(model, dummy)
        if menu == '4':
            pass