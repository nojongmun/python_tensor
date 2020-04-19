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
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

class Titanic:
    def __init__(self):
        # 지도학습구조에 정형적인 5개
        # 디렉토리와 파일을 나눠서 본다.
        self._context = None
        self._fname = None
        self._train = None
        self._test = None
        self._test_id = None

    @property
    def context(self) -> str:
        return self._context

    @context.setter
    def context(self, context):
        self._context = context

    @property
    def fname(self) -> str:
        return self._fname

    @fname.setter
    def fname(self, fname):
        self._fname = fname

    @property
    def train(self) -> str:
        return self._frain

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
    def test_id(self) -> str:
        return self._test_id

    @test_id.setter
    def test_id(self, test_id):
        self._test_id = test_id

    # 기본적으로 사용되는 코드
    def new_file(self) -> str: return self._context + self._fname
    def new_dframe(self) -> object: return pd.read_csv(self.new_file())

    """Index(['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age',
       'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],   dtype='object')"""

    def hook_process(self, train, test):
        print('------- 1. Cabin & Ticket 삭제 -----------')
        lst = self.drop_feature(train, test, 'Cabin')
        lst = self.drop_feature(lst[0], lst[1], 'Ticket')
        print(lst[0].columns)

        print('------- 2. Embarked Nominal  -----------')
        lst = self.embarked_nominal(lst[0], lst[1])
        print('------- 3. Sex Nominal  -----------')
        lst = self.sex_nominal(lst[0], lst[1])

        print(lst[0]['Embarked'].head())
        print(lst[0]['Sex'].head())

        print('------- 4. Fare ordinal  -----------')
        lst = self.fare_ordinal(lst[0], lst[1])
        print(lst[0]['FareBand'].head())

        print('------- 5. Title nominal  -----------')
        lst = self.title_nominal(lst[0], lst[1])
        print(lst[0]['Title'].head())
        print('------- 6. Drop PassengerId ,Name , Fare -----------')
        self._test_id = test['PassengerId']
        lst = self.drop_feature(lst[0], lst[1], 'PassengerId')
        lst = self.drop_feature(lst[0], lst[1], 'Name')
        print(lst[0].columns)
        lst = self.drop_feature(lst[0], lst[1], 'Fare')
        print(lst[0].columns)
        print('------- 7. Drop Ordinal -----------')
        lst = self.age_ordinal(lst[0], lst[1])
        print(lst[0]['AgeGroup'].head())
        print('------- 8. Final null check-----------')
        null_sum = self.null_sum(lst[0])
        print('train null count {} ' .format(null_sum))
        null_sum = self.null_sum(lst[1])
        print('test null count {} ' .format(null_sum))
        lst[1].fillna({'FareBand': 1})
        self._test = lst[1]
        print('------- 9. Drop Survived-----------')
        # train = lst[0].drop(["Survived"], axis=1)
        return lst[0]

    # 자료를 삭제할때는 반드시 두자료를 같이 삭제
    @staticmethod
    def drop_feature(train, test, feature) -> []:
        train = train.drop([feature], axis=1)  # 세로방향 =>1, 가료방향=>0
        test = test.drop([feature], axis=1)  # 세로방향 =>1, 가료방향=>0
        return [train, test]

    # 항구 편집(항구문자를 숫자롤 표현) Nominal 처리
    @staticmethod
    def embarked_nominal(train, test) -> []:
        train = train.fillna({"Embarked":"S"})
        city_mapping = {"S": 1, "C": 2, "Q": 3}
        train['Embarked'] = train['Embarked'].map(city_mapping)
        test['Embarked'] = test['Embarked'].map(city_mapping)
        return [train, test]

    @staticmethod
    def sex_nominal(train, test) ->[]:
        sex_mapping = {"male": 0, "female": 1}
        # train['Sex'] = train['Sex'].map(sex_mapping)
        # test['Sex'] = test['Sex'].map(sex_mapping)
        combine = [train, test]
        for dataset in combine:
            dataset['Sex'] = dataset['Sex'].map(sex_mapping)
        return [train, test]

    @staticmethod
    def fare_ordinal(train, test) ->[]:
        # 균등하계 4등급으로 나눈다.
        train['FareBand'] = pd.qcut(train['Fare'], 4, labels=[1,2,3,4])
        test['FareBand'] = pd.qcut(test['Fare'], 4, labels=[1, 2, 3, 4])
        return [train, test]

    @staticmethod
    def title_nominal(train, test) ->[]:
        combine = [train, test]
        for dataset in combine:
            # 정규표현식 : \.=> 자연어 (.)점을 말함 , expand=False => 나머지는 버리자
            dataset['Title'] = dataset.Name.str.extract('([A-Za-z])\.', expand=False)
        for dataset in combine:
            dataset['Title'] = dataset['Title'].replace(['Capt', 'Col', 'Don', 'Dr', 'Major', 'Rev', 'Jonkheer','Dona'], 'Rare')
            dataset['Title'] = dataset['Title'].replace(['Countess', 'Lady', 'Sir'], 'Royal')
            dataset['Title'] = dataset['Title'].replace(['Mile', 'Ms'], 'Miss')

        title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Royal": 5, "Rare": 6, "Mne": 7}
        for dataset in combine:
            dataset['Title'] = dataset['Title'].map(title_mapping)
            dataset['Title'] = dataset['Title'].fillna(0)
        return [train, test]

    @staticmethod
    def age_ordinal(train, test) ->[]:
        train['Age'] = train['Age'].fillna(-0.5)
        test['Age'] = test['Age'].fillna(-0.5)
        # np.inf => 기타 값 처리
        bins = [-1, 0, 5, 12, 18, 24, 35, 60, np.inf]
        labels = ['Unknown', 'Baby', 'Child', 'Teenager','Student','Young Adult','Adult','Senior']
        train['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        test['AgeGroup'] = pd.cut(train['Age'], bins, labels=labels)
        age_title_mapping = {0: "Unknown",  1: "Baby", 2: "Child",  3: "Teenager",  4: "Student", 5: "Young Adult", 6: "Adult", 7: "Senior"}
        for x in range(len(train['AgeGroup'])):
            if train['AgeGroup'][x] == 'Unknown' :
                train['AgeGroup'][x] = age_title_mapping[train['Title'][x]]
        for x in range(len(test['AgeGroup'])):
            if test['AgeGroup'][x] == 'Unknown':
                test['AgeGroup'][x] = age_title_mapping[test['Title'][x]]
        age_mapping = {"Unknown": 0, "Baby": 1, "Child": 2, "Teenager": 3, "Student": 4, "Young Adult": 5, "Adult": 6, "Senior": 7}
        train['AgeGroup'] = train['AgeGroup'].map(age_mapping)
        test['AgeGroup'] = test['AgeGroup'].map(age_mapping)
        return [train, test]

    @staticmethod
    def null_sum(train) -> int:
        return train.isnull().sum()

    def hook_test(self, model, dummy):
        print('결정트리 검증 정확도 {} % '. format(self.accuracy_by_dtree(model, dummy)))
        print('랜덤프렝스트 검증 정확도 {} % '.format(self.accuracy_by_rforest(model, dummy)))
        print('KNN 검증 정확도 {} % '.format(self.accuracy_by_knn(model, dummy)))
        print('나이브 베이즈 검증 정확도 {} % '.format(self.accuracy_by_nb(model, dummy)))
        print('SVM 정확도 {} % '.format(self.accuracy_by_svm(model, dummy)))

    @staticmethod
    def create_k_fold():
        return KFold(n_splits=10, shuffle=True, random_state=0)

    @staticmethod
    def create_random_variable(train, X_feature, Y_features) ->[]:
        the_X_feature = X_feature
        the_Y_feature = Y_features
        train2,  test2 = train_test_split(train, test_size=0.3, random_state=0)
        train_X = train2[the_X_feature]
        train_Y = train2[the_Y_feature]
        test_X = test2[the_X_feature]
        test_Y = test2[the_Y_feature]
        return [train_X, train_Y, test_X, test_Y]

    def accuracy_by_dtree(self, model, dummy):
        clf = DecisionTreeClassifier()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_rforest(self, model, dummy):
        clf = RandomForestClassifier()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_nb(self, model, dummy):
        clf = GaussianNB()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_knn(self, model, dummy):
        clf = KNeighborsClassifier(n_neighbors=13)
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

    def accuracy_by_svm(self, model, dummy):
        clf = SVC()
        scoring = 'accuracy'
        k_fold = self.create_k_fold()
        score = cross_val_score(clf, model, dummy, cv=k_fold, n_jobs=1, scoring=scoring)
        accuracy = round(np.mean(score) * 100, 2)
        return accuracy

