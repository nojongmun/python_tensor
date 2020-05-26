from flask import Flask
from flask import render_template, request, jsonify
from model.ai_service import AiService
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/cabbage', methods=['POST'])
def cabbage():
    ai = AiService()
    result = ai.service()  # 이름을 맞춰주면 자동으로 들어온다.
    render_params = {}
    render_params['result'] = result
    return render_template('index.html', **render_params)


if __name__ == '__main__':
    app.debug = True
    app.run()