import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
from tensorflow.keras.models import load_model
from sklearn import datasets
# pip uninstall keras
# pip install keras==2.2.5 // 버전처리 안됨
from flask_restful import reqparse
from tensorflow.keras import backend as K

class AiService:
    def __init__(self):
        global model, graph, target_names
        K.clear_session()
        model = load_model('model/saved_model/iris.h5')
        graph = tf.get_default_graph()
        target_names = datasets.load_iris().target_names

    # sepal_length = 5.1 & sepal_width = 3.5 & petal_length = 1.4 & petal_width = 0.2
    def service_model(self):
        parser = reqparse.RequestParser()
        parser.add_argument('sepal_length', type=float)
        parser.add_argument('sepal_width', type=float)
        parser.add_argument('petal_length', type=float)
        parser.add_argument('petal_width', type=float)
        args = parser.parse_args()
        features = [args['sepal_length'],
                    args['sepal_width'],
                    args['petal_length'],
                    args['petal_width']]
        features = np.reshape(features, (1, 4))
        with graph.as_default():
            Y_pred = model.predict_classes(features)
        result = {'species' : target_names[Y_pred[0]]}
        return result

# if __name__ == '__main__':
#    print('케라스 버전')
#    print(tf.keras.__version__)
#    print('텐서 버전')
#    print(tf.__version__)