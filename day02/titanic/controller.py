from titanic.model import Titanic
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class Controller:
    def __init__(self):
        self._titanic = Titanic()
        self._context = './data/'
        self._train = self.create_train()

    def create_train(self) -> object:
        titanic = self._titanic
        titanic.context = self._context
        titanic.fname = 'train.csv'
        train = titanic.new_dframe()
        titanic.fname = 'test.csv'
        test = titanic.new_dframe()
        return titanic.hook_process(train, test)

    @staticmethod
    def create_model(train) -> object:
        return train.drop('Survived', axis=1)

    @staticmethod
    def create_dummy(train) -> object:
        return train['Survived']

    def test_all(self, model, dummy):
        titanic = self._titanic
        titanic.hook_test(model, dummy)

    def submit(self, train):
        titanic = self._titanic
        model = self.create_model()
        dummy = self.create_dummy()
        test = titanic.test
        test_id = titanic.test_id
        clf = RandomForestClassifier()
        clf.fit(model, dummy)
        prediction = clf.predict(test)
        submission = pd.DataFrame({'PassengerId': test_id, 'Survived':prediction})
        submission.to_csv('./data/submission.csv', index=False)
