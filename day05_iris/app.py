from flask import Flask
from flask import render_template, request, jsonify
from model.member_service import MemberService
from model.ai_service import AiService

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login', methods=['POST'])
def login():
   userid = request.form['userid']
   passwrod = request.form['password']
   print('입력된 아이디 : {}'.format(userid))
   service = MemberService()
   # service.create_table()  # 처음 한번 실행하고 주석처리 하자

   view = service.login(userid, passwrod)
   return render_template(view)

@app.route('/iris', methods=['POST'])
def iris():
    service = AiService()
    result = service.service_model()
    return render_template('iris.html', result=result)

if __name__ == '__main__':
    app.debug = True
    app.run()