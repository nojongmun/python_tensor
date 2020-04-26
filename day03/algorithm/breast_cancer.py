from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
class BreastCancer:
    def create_caner(self):
        return load_breast_cancer()
    def predict_by_decision_tree(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=42 )
        classfier = DecisionTreeClassifier(random_state=0)
        classfier.fit(X_train, y_train)
        print('Decision Tree 훈련세트 정확도 : {:.3f}'.format(classfier.score(X_train, y_train)))
        print('Decision Tree 테스트 세트 정확도 : {:.3f}'.format(classfier.score(X_test, y_test)))
    def predict_by_random_forest(self):
        cancer = load_breast_cancer()
        X_train, X_test, y_train, y_test = train_test_split(
            cancer.data, cancer.target, stratify=cancer.target, random_state=42)
        classfier = RandomForestClassifier(n_estimators=100, random_state=0)
        classfier.fit(X_train, y_train)
        print('RandomForest 훈련세트 정확도 : {:.3f}'.format(classfier.score(X_train, y_train)))
        print('RandomForest 테스트 세트 정확도 : {:.3f}'.format(classfier.score(X_test, y_test)))
        # 특성 중요도
        print('RandomForest 특성 중요도 : {}'.format(classfier.feature_importances_))
        return classfier
    def plot_feature_importances(self, classfier):
        cancer = load_breast_cancer()
        n_features = cancer.data.shape[1]
        plt.barh(range(n_features), classfier.feature_importances_, align='center')
        plt.yticks(np.arange(n_features), cancer.feature_names)
        plt.xlabel('attr importances')
        plt.ylabel('attr')
        plt.ylim(-1, n_features)
        plt.show()
if __name__ == '__main__':
    this = BreastCancer()
    # this.predict_by_decision_tree()
    classfier = this.predict_by_random_forest()
    this.plot_feature_importances(classfier)