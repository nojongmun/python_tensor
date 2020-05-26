from flask import Flask
from flask import render_template, request, jsonify
from model.blood import Model
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/blood', methods=['POST'])
def blood():
    # 파라미터값 받기
    weight = request.form['weight']
    age = request.form['age']

    print('넘어온 몸무게 : {}, 나이: {}'.format(weight,age))
    # 인공지능에게 주입
    model = Model('model/data/data.txt')
    raw_date = model.create_raw_data()
    render_params = {}
    value = model.create_model(raw_date, weight, age)
    print('AI가 예층한 혈중농동 : {}'.format(value))
    render_params['result'] = value
    return render_template('index.html', **render_params)


if __name__ == '__main__':
    app.debug = True
    app.run()