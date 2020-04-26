"""
PassengerId 테스트 시 문제로 제공됨
survival	생존여부	0 = No, 1 = Yespclass
승선권	1 = 1st, 2 = 2nd, 3 = 3rd
sex  성별
Age 나이
name 이름
sibsp 동반한 형제, 자매, 배우자
parch	동반한 부모,자식
ticket	티켓번호
fare	티켓의 요금
cabin	객실번호
embarked	승선한 항구명
C = 쉐부로, Q = 퀸즈타운, S = 사우스햄톤
Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
       'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],   dtype='object')
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import  matplotlib.pyplot as plt
import seaborn as sns
import  pandas as pd
class Titanic:
    context: str
    fname: str
    train: object
    test: object
    id: str
    model: object
    label: object

    @property
    def context(self) -> str: return self._context
    @context.setter
    def context(self, context): self._context = context
    @property
    def fname(self) -> str: return self._fname
    @fname.setter
    def fname(self, fname): self._fname = fname
    @property
    def train(self) -> str: return self._train
    @train.setter
    def train(self, train):
        self._train = train

    @property
    def test(self) -> str:
        return self._test

    @test.setter
    def test(self, test):
        self._test = test

    @property
    def id(self) -> str:
        return self._id

    @id.setter
    def id(self, id):
        self._id = id

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, label):
        self._label = label

    @property
    def model(self) -> str:
        return self._model

    @model.setter
    def model(self, model):
        self._model = model

    def new_file(self) -> str:
        return self._context + self._fname

    def new_dframe(self) -> object:
        return pd.read_csv(self.new_file())

    def modeling(self, this):
        print('---------- 1. Drop PassengerId, Cabin, Ticket  ------------')
        this = self.drop_feature(this, 'PassengerId')
        this = self.drop_feature(this, 'Cabin')
        this = self.drop_feature(this, 'Ticket')
        print('---------- 2. Embarked, Sex Nominal ------------')
        this = self.embarked_nominal(this)
        this = self.sex_nominal(this)
        print('---------- 3. Fare Ordinal ------------')
        this = self.fare_ordinal(this)
        this = self.drop_feature(this, 'Fare')
        print('---------- 4. Title Nominal ------------')
        this = self.title_nominal(this)
        this = self.drop_feature(this, 'Name')
        print('---------- 5. Age Ordinal ------------')
        this = self.age_ordinal(this)
        print('---------- 6. Final Null Check ------------')
        print('train null count \n{}'.format(this.train.isnull().sum()))
        print('test null count \n{}'.format(this.test.isnull().sum()))
        print('---------- 7. Create Model ------------')
        self.model = this.train.drop('Survived', axis=1)
        self.label = this.train['Survived']
        return this

    @staticmethod
    def age_ordinal(this) -> []:
        train = this.train
        test = this.test
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager', 'Student', 'Young Adult', 'Adult', 'Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(test['Age'], bins, labels=labels)
        age_title_mappeing = {
            0: 'Unknown', 1: 'Baby', 2: 'Child', 3: 'Teenager', 4: 'Student', 5: 'Young Adult', 6: 'Adult', 7: 'Senior'
        }
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown':
                train['AgeGroup'][x] = age_title_mappeing[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mappeing[test['Title'][x]]
        age_mapping = {
            'Unknown': 0, 'Baby': 1, 'Child': 2, 'Teenager': 3, 'Student': 4, 'Young Adult': 5, 'Adult': 6, 'Senior': 7
        }
        this.train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        this.test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        return this

    @staticmethod
    def title_nominal(this) -> []:
        combine = [this.train, this.test]
        for dataset in combine:
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z])\.', expand=False)
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(
                ['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer', 'Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            dataset['Title'] = dataset['Title'].replace(['Mile', 'Ms'], 'Miss')
        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6, "Mne": 7}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        return this

    @staticmethod
    def fare_ordinal(this) -> []:
        this.train['FareBand'] = pd.qcut(this.train['Fare'], 4, labels=[1, 2, 3, 4])
        this.test['FareBand'] = pd.qcut(this.test['Fare'], 4, labels=[1, 2, 3, 4])
        this.train = this.train.fillna({'FareBand': 1})
        this.test = this.test.fillna({'FareBand': 1})
        return this

    @staticmethod
    def drop_feature(this, feature) -> []:
        this.train = this.train.drop([feature], axis=1)
        this.test = this.test.drop([feature], axis=1)
        return this

    @staticmethod
    def embarked_nominal(this) -> []:
        this.train = this.train.fillna({"Embarked": "S"})
        city_mapping = {"S": 1, "C": 2, "Q": 3}
        this.train['Embarked'] = this.train['Embarked'].map(city_mapping)
        this.test['Embarked'] = this.test['Embarked'].map(city_mapping)
        return this

    @staticmethod
    def sex_nominal(this) -> []:
        sex_mapping = {"male": 0, "female": 1}
        combine = [this.train, this.test]
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        return this

    def learning(self, this):
        print('결정트리 검증 정확도 {} % '.format(self.calculate_accuracy(this, DecisionTreeClassifier())))
        print('랜덤프레스트 검증 정확도 {} % '.format(self.calculate_accuracy(this, RandomForestClassifier())))
        print('KNN 검증 정확도 {} % '.format(self.calculate_accuracy(this, KNeighborsClassifier())))
        print('나이브 베이즈 검증 정확도 {} % '.format(self.calculate_accuracy(this, GaussianNB())))
        print('SVM 검증 정확도 {} % '.format(self.calculate_accuracy(this, SVC())))

    @staticmethod
    def calculate_accuracy(this, classfier):
        score = cross_val_score(classfier,
                                this.model,
                                this.label,
                                cv=KFold(n_splits=10, shuffle=True, random_state=0),
                                n_jobs=1,
                                scoring='accuracy')
        return round(np.mean(score) * 100, 2)

class View:
    def __init__(self):
        self._titanic = Titanic()
        self._context = './data/'

    def create_train(self) -> object:
        titanic = self._titanic
        titanic.context = self._context
        titanic.fname = 'train.csv'
        return titanic.new_dframe()

    @staticmethod
    def plot_survivled_dead(train):

        f, ax = plt.subplots(1, 2,  figsize=(18, 8))
        train['Survived'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f', ax=ax[0], shadow=True)
        ax[0].set_title('Survived')
        ax[1].set_ylabel('')
        sns.countplot('Survived', data=train, ax=ax[1])
        ax[1].set_title('Survived')
        plt.show()

    @staticmethod
    def plot_sex(train):
        f, ax = plt.subplots(1, 2, figsize=(18, 8))
        train['Survived'][train['Sex'] == 'male'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f', ax=ax[0], shadow=True)
        train['Survived'][train['Sex'] == 'female'].value_counts().plot.pie(explode=[0, 0.1], autopct='%1.1f', ax=ax[1], shadow=True)
        ax[0].set_title('Male')
        ax[1].set_title('Female')
        plt.show()

    @staticmethod
    def bar_chart(train, feature):
        survived = train[train['Survived'] == 1][feature].value_counts()
        dead = train[train['Survived'] == 0][feature].value_counts()
        df = pd.DataFrame([survived, dead])
        df.index = ['survived', 'dead']
        df.plot(kind='bar', stacked=True, figsize=(110,7))
        plt.show()

if __name__ == '__main__':
    def print_menu():
        print('0.  Exit')
        print('1.  Create Model')
        print('2.  Data Visualize')
        print('3.  Modeling')
        print('4.  Learning')
        print('5.  Submit')
        return input('Choose One\n')
    this = Titanic()
    view = View()

    while 1:
        menu = print_menu()
        print('Menu : %s '  % menu)
        if menu == '0':
            print('stop')
            break;
        if menu == '1':
            this.context = './data/'
            this.fname = 'train.csv'
            this.new_file()
            this.train = this.new_dframe()
            this.fname = 'test.csv'
            this.new_file()
            this.test = this.new_dframe()
            this.id = this.test['PassengerId']
        if menu == '2':
            view = View()
            temp = view.create_train()
            view.plot_survivled_dead(temp)
            # view.plot_sex(temp)
            # view.bar_chart(temp, 'Pclass')
        if menu == '3':
            this = this.modeling(this)
        if menu == '4':
            this.learning(this)
        if menu == '5':
            classfier = RandomForestClassifier()
            classfier.fit(this.model, this.label)
            prediction = classfier.predict(this.test)
            submission = pd.DataFrame(
                {'Passengerid': this.id, 'Survived':prediction}
            )
            submission.to_csv('./data/submission.csv', index=False)