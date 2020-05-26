# db 연결
import _sqlite3

class Model:
    def __init__(self):
        self.conn = _sqlite3.connect('sqlite.db')

    def create(self):
        query = """
            create table if not exists member(
                userid varchar(10) primary key,
                password varchar(10),
                phone varchar(15),
                regdate date default current_timestamp
            )
        """
        self.conn.execute(query)
        self.conn.commit()

    def insert_many(self):
        data = [('lee','1', '010-9999-9999'), ('kim','1', '010-9999-1234'), ('park','1', '010-1234-9999')]
        query = """
            insert into member(userid, password, phone) values (?, ?, ?)
        """
        self.conn.executemany(query, data)
        self.conn.commit()

    def fetch_one(self):
        query = """
                    select * from member where userid like 'lee'
                """
        cursor = self.conn.execute(query)
        one = cursor.fetchone()
        print('검색 결과: {}'.format(one))

    def fetch_all(self):
        query = """
                    select * from member
                """
        cursor = self.conn.execute(query)
        all = cursor.fetchall()
        count = 0
        for i in all:
            count += 1
        print('총 인원 {} 명'.format(count))

    def login(self, userid, password):
        query = """
                select * from member where userid like ?  and password like ?
         """
        data = [userid, password]
        cursor = self.conn.execute(query, data)
        one = cursor.fetchone()
        print('로그인 회원정보 : {}'.format(one))
        return one