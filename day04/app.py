# https://parksrazor.tistory.com/1
# app.py 만들기 => templates 폴더 만들기 =>  폴더 안에  index.html 만든다.
# https://parksrazor.tistory.com/99
from flask import Flask
from flask import render_template, request, jsonify
import re

app = Flask(__name__)

# 처음 실행해서 확인
# @app.route('/')
# def hello_world():
#    return 'hello world!'

# @app.route("/move/<path>")
# def move(path):
#    return render_template('{}.html'.format(path))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/move/calculator")
def move():
    return render_template('calculator.html')

@app.route("/calc", methods=["POST"])
def calc():
    # 파라미터값 받기
    num1 = request.form['num1']
    num2 = request.form['num2']
    opcode = request.form['opcode']
    # print('넘어온 num1 값 : {}'.format(num1))
    # print('넘어온 num2 값 : {}'.format(num2))
    # print('넘어온 연산자 : {}'.format(opcode))
    if opcode == "add":
        result = int(num1) + int(num2)
    elif opcode == "sub":
        result = int(num1) - int(num2)
    elif opcode == "mul":
        result = int(num1) * int(num2)
    elif opcode == "div":
        result = int(num1) / int(num2)

    render_params = {}
    render_params['result'] = int(result)
    # 결과값을 가지고 가라
    return render_template('calculator.html', **render_params)

if __name__ == '__main__':
    app.debug = True
    app.run()