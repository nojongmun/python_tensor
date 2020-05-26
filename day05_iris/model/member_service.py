from model.member import Model

class MemberService:
    def __init__(self):
        self.model = Model()

    def create_table(self):
        self.model.create()
        self.model.insert_many()
        self.model.fetch_all()
        self.model.fetch_one()

    def login(self, userid, password):
        one = self.model.login(userid, password)
        if one is None:
            view = 'index.html'
        else:
            view = 'iris.html'
        return view

'''
한번만 실행하면 됨
if __name__ == '__main__':
    service = MemberService()
    service.create_table()
'''