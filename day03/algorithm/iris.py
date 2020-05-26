# https://parksrazor.tistory.com/83
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
import tensorflow as tf


# 지도학습할때 사용하는 패턴
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

class Perceptron:
    def __init__(self, eta = 0.01, n_iter = 50, random_state = 1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    # X =>  들어가는 데이터는 무지 많음(확률변수) => 무조건 대문자 X
    def fit(self, X, y):
        regen = np.random.RandomState(self.random_state)
        self.w_ = regen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.n_iter):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[1:] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class Adaline:
    def __init__(self, eta=0.01, n_iter=50, random_state=None, shuffle=True):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state
        self.shuffle = shuffle

    def _shuffle(self, X, y):
        r = self.rgen.permutation(len(y))
        return X[r], y[r]

    def _initialize_weights(self, m):
        """랜덤한 작은 수로 가중치를 초기화"""
        self.rgen = np.random.RandomState(self.random_state)
        self.w_ = self.rgen.normal(loc=0.0, scale=0.01, size=1 + m)
        self.w_initialized = True

    def activation(self, X):
        return X  # 선형활성화

    def _update_weights(self, xi, target):
        """ 아달린 학습 규칙을 적용하기 위해 가중치 업데이트 함"""
        # eta : 학습률 0.0 ~ 1.0
        output = self.activation(self.net_input(xi))
        error = (target - output)
        self.w_[1:] += self.eta * xi.dot(error)
        self.w_[0] += self.eta * error
        cost = 0.5 * error ** 2
        return cost

    def fit(self, X, y):
        self._initialize_weights(X.shape[1])
        self.cost_ = []
        for _ in range(self.n_iter):
            if self.shuffle:
                X, y = self._shuffle(X, y)
            cost = []
            for xi, target in zip(X, y):
                cost.append(self._update_weights(xi, target))
            avg_cost = sum(cost) / len(y)
            self.cost_.append(avg_cost)
        return self

    def partial_fit(self, X, y):
        """가중치를 다시 초기화 하지 않고 훈련 데이터를 학습"""
        if not self.w_initialized:
            self._initialize_weights(X.shape[1])
        if y.ravel().shape[0] > 1:
            for xi, target in zip(X, y):
                self._update_weights(xi, target)
        else:
            self._update_weights(X, y)
        return self

    def net_input(self, X):
        """최종입력계산"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.01, 1, -1)

class Iris:
    iris: object

    def __init__(self):
        self.storage = Storage()
        self.neural_network = None

    def initialize(self):
        self.iris = pd.read_csv('https://archive.ics.uci.edu/ml/'
                                'machine-learning-databases/iris/iris.data', header=None)
        print(self.iris.head())
        print(self.iris.columns)
        """
        setosa, versicolor, virginica의 세가지 붓꽃 종(species)
        feature
        꽃받침 길이(Sepal Length)
        꽃받침 폭(Sepal Width)
        꽃잎 길이(Petal Length)
        꽃잎 폭(Petal Width)
        """
        # setosa 와 versicolor 선택
        temp = self.iris.iloc[0:100, 4].values
        this = self.storage
        this.train_Y = np.where(temp == 'Iris-setosa', -1, 1)
        # 꽃받침 길이와 꽃잎 폭 선택
        this.train_X = self.iris.iloc[0:100, [0, 2]].values
        self.neural_network = Perceptron(eta=0.1, n_iter=10)


    def get_iris(self): return self.iris
    def get_X(self): return self._X
    def get_y(self): return self._y
    ''' 
    단층 퍼셉트론 
    '''
    def draw_scatter(self):
        X = self.X
        plt.scatter(X[:50, 0], X[:50, 1], color = 'red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1], color = 'blue', marker='x', label='versi')
        plt.xlabel('sepal length[cm]')
        plt.ylabel('petal length[cm]')
        plt.legend(loc='upper left')
        plt.show()

    def draw_errors(self):
        X = self.X
        y = self.y
        self.classfier.fit(X,y)
        plt.plot(range(1, len(self.classfier.errors_) + 1), self.classfier.errors_, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Number of errors')
        plt.show()

    # 결정 경계
    def plot_decision_regions(self):
        X = self.X
        y = self.y
        classfier = Perceptron(eta=0.1, n_iter=4)
        classfier.fit(X, y)
        colors = ('red','blue','lightgreen','gray','cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        resolution = 0.2
        """
        numpy 모듈의 arange 함수는 반열린구간 [start, stop) 에서 
        step 의 크기만큼 일정하게 떨어져 있는 숫자들을 
        array 형태로 반환해 주는 함수
        meshgrid 명령은 사각형 영역을 구성하는 
        가로축의 점들과 세로축의 점을 
        나타내는 두 벡터를 인수로 받아서 
        이 사각형 영역을 이루는 조합을 출력한다. 
        결과는 그리드 포인트의 x 값만을 표시하는 행렬과
        y 값만을 표시하는 행렬 두 개로 분리하여 출력한
        """
        xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution), np.arange(x2_min, x2_max, resolution))
        Z = classfier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contour(xx1, xx2, Z, alpha=0.3, cmap = cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())
        # 샘플 산점도
        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x = X[y==cl, 0], y = X[y==cl, 1], alpha=0.8, c=colors[idx], label = cl, edgecolors='black')
        plt.xlabel('sepal length[cm]')
        plt.ylabel('petal length[cm]')
        plt.legend(loc='upper left')
        plt.show()

    ''' 
    멀티 퍼셉트론 (MLP) 
    '''
    def show_scatter(self):
        this = self.storage
        X = this.train_X
        plt.scatter(X[:50, 0], X[:50, 1],
                    color='red', marker='o', label='setosa')
        plt.scatter(X[50:100, 0], X[50:100, 1],
                    color='blue', marker='x', label='versicolor')
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(loc='upper left')
        plt.show()

    def show_errors(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y
        self.neural_network.fit(X, y)
        plt.plot(range(1, len(self.neural_network.errors_) + 1),
                 self.neural_network.errors_, marker='o')
        plt.xlabel('Epoch')
        plt.ylabel('Number of Errors')
        plt.show()

    def show_decision_tree(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y
        nn = self.neural_network
        nn.fit(X, y)
        colors = ('red', 'blue', 'rightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])
        x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1,
        x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1,
        resolution = 0.2
        """
        numpy 모듈의 arange 함수는 반열린 구간 [start, stop] 에서
        step 의 크기만큼 일정하게 떨어져 있는 숫자들을 array 형태로 반환해 주는 함수

        meshgrid 명령은 사각형 영역을 구성하는 
        가로축의 점들과 세로축의 점을 나타내는 두 벡터를 인수로 받아서
        이 사각형 영역을 이루는 조합을 출력한다
        결과는 그리드 포인트의 x 값만을 표시하는 행렬과
        y값 만을 표시하는 행렬 두 개로 불리하여 출력한다.
        """
        xx1, xx2 = np.meshgrid(
            np.arange(x1_min, x1_max, resolution),
            np.arange(x2_min, x2_max, resolution),
        )
        Z = nn.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)
        plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y)):
            plt.scatter(x=X[y == cl, 0],
                        y=X[y == cl, 1],
                        alpha=0.8,
                        c=colors[idx], label=cl, edgecolors='black')

        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.legend(loc='upper left')
        plt.show()

    def show_adaline(self):
        this = self.storage
        X = this.train_X
        y = this.train_Y
        X_std = np.copy(X)
        X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
        X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
        self.neural_network = Adaline(eta=0.01, n_iter=50, random_state=1)
        self.neural_network.fit(X_std, y)
        self.show_decision_tree()

'''
if __name__ == '__main__':
    
    this = Iris()
    this.get_iris()
    # this.draw_scatter()
    # this.draw_errors()
    this.plot_decision_regions()
'''
if __name__ == '__main__':
    iris = Iris()
    iris.initialize()
    # iris.show_scatter()
    # iris.show_errors()
    iris.show_decision_tree()

























