# https://parksrazor.tistory.com/85
from tensorflow.keras.datasets import boston_housing
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt

# 지도학습 코드 패턴
class Storage:
    train_X: object
    train_Y: object
    test_X: object
    test_Y: object

    @property
    def train_X(self) -> object: return self._train_X

    @train_X.setter
    def train_X(self, train_X): self._train_X = train_X

    @property
    def train_Y(self) -> object: return self._train_Y

    @train_Y.setter
    def train_Y(self, train_Y): self._train_Y = train_Y

    @property
    def test_X(self) -> object: return self._test_X

    @test_X.setter
    def test_X(self, test_X): self._test_X = test_X

    @property
    def test_Y(self) -> object: return self._test_Y

    @test_Y.setter
    def test_Y(self, test_Y): self._test_Y = test_Y


class Boston:
    boston: object
    def __init__(self):
        self.storage = Storage()

    @property
    def boston(self) -> object: return self._boston

    @boston.setter
    def boston(self, boston): self._boston = boston

    def initialize(self):
        this = self.storage
        (this.train_X, this.train_Y), (this.test_X, this.test_Y) = boston_housing.load_data()
        print(f'확률변수 X의 길이 : {len(this.train_X)}')
        print(f'확률변수 Y의 길이 : {len(this.train_Y)}')
        print(f'확률변수 X[0] : {this.train_X[0]}')
        print(f'확률변수 Y[0] : {this.train_Y[0]}')

    def standardization(self):  # 데이터 전처리 .. 정규화
        this = self.storage
        x_mean = this.train_X.mean()
        x_std = this.train_X.std()
        this.train_X -= x_mean
        this.train_X /= x_std
        this.test_X -= x_mean
        this.test_X /= x_std
        y_mean = this.train_Y.mean()
        y_std = this.train_Y.std()
        this.train_Y -= y_mean
        this.train_Y /= y_std
        this.test_Y -= y_mean
        this.test_Y /= y_std

    def new_model(self):
        model = keras.Sequential([
            keras.layers.Flatten(),
            keras.layers.Dense(units=52, activation='relu', input_shape=(13, 1)),
            keras.layers.Dense(units=39, activation='relu'),
            keras.layers.Dense(units=26, activation='relu'),
            keras.layers.Dense(units=1)
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=0.07), loss='mse')

        return model
    # 학습
    def learn_model(self, model):
        this = self.storage
        history = model.fit(this.train_X, this.train_Y, epochs=25,
                            batch_size=32, validation_split=0.25,
                            callbacks=[tf.keras.callbacks.EarlyStopping(patience=3,
                                                                        monitor='val_loss')])
        return history
    # 시험
    def eval_model(self, model):
        this = self.storage
        print(model.evaluate(this.test_X, this.test_Y))
        return this

def show_history(history):
    plt.plot(history.history['loss'], 'b-', label='loss')
    plt.plot(history.history['val_loss'], 'r--', label='val_loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

def show_boston(dic):
    model = dic['model']
    this = dic['storage']
    test_X = this.test_X
    test_Y = this.test_Y
    pred_Y = model.predict(test_X)
    plt.figure(figsize=(5, 5))
    plt.plot(this.test_Y, pred_Y, 'b.')
    plt.axis([min(this.test_Y), max(test_Y), min(test_Y), max(test_Y)])
    plt.plot([min(test_Y), max(test_Y)], [min(test_Y), max(test_Y)], ls="--", c=".3")
    plt.xlabel('test_Y')
    plt.ylabel('pred_Y')
    plt.show()

if __name__ == '__main__':
    boston = Boston()
    boston.initialize()
    boston.standardization()
    model = boston.new_model()
    history = boston.learn_model(model)
    storage = boston.eval_model(model)
    # show_history(history)
    show_boston({'model': model, 'storage': storage})