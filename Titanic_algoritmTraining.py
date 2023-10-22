''' Model Selection - Titanic Challange '''

#  Import Libraries

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import pickle

from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score
from yellowbrick.classifier import ConfusionMatrix

from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

with open('titanic.pkl', mode='rb') as f:
    X_treinamento, y_treinamento, X_test, y_test = pickle.load(f)



#  Nayve Bayes
print('*** Nayve Baes - Score ***')

nayve_train = GaussianNB()
nayve_train.fit(X_treinamento, y_treinamento)
nayve_previsao = nayve_train.predict(X_test)

print(accuracy_score(y_test, nayve_previsao))


#  Decision Tree
print('*** Decision Tree - Score ***')

decisionTree_train = DecisionTreeClassifier(criterion='entropy')
decisionTree_train.fit(X_treinamento, y_treinamento)
decisionTree_previsao = decisionTree_train.predict(X_test)

print(accuracy_score(y_test, decisionTree_previsao))

# #       Confusion Matrix
# cm = ConfusionMatrix(decisionTree_train)
# cm.fit(X_treinamento, y_treinamento)
# cm.score(X_test, y_test)


#  Random Forest
print('*** Random Forest - Score ***')

rndForest_train = RandomForestClassifier(n_estimators=60,
                                         criterion='entropy',
                                         random_state=0)
rndForest_train.fit(X_treinamento, y_treinamento)
rndForest_previsao = rndForest_train.predict(X_test)

print(accuracy_score(y_test, rndForest_previsao))


#  kNN
print('*** kNN - Score ***')

kNN_train = KNeighborsClassifier(n_neighbors=12,
                                 metric='minkowski',
                                 p=2)
kNN_train.fit(X_treinamento, y_treinamento)
kNN_previsao = kNN_train.predict(X_test)

print(accuracy_score(y_test, kNN_previsao))

#  regressLogistica
print('*** regressLogistica - Score ***')

regressLogistica_train = LogisticRegression(random_state=1)
regressLogistica_train.fit(X_treinamento, y_treinamento)
regressLogistica_previsao = regressLogistica_train.predict(X_test)

print(accuracy_score(y_test, regressLogistica_previsao))


#  SVM
print('*** SVM - Score ***')

SVM_train = SVC(kernel='rbf',
                random_state=1,
                C=2.0)
SVM_train.fit(X_treinamento, y_treinamento)
SVM_previsao = SVM_train.predict(X_test)

print(accuracy_score(y_test, SVM_previsao))


#  neuralNetwork
print('*** neuralNetwork - Score ***')

neuralNetwork_train = MLPClassifier(max_iter=1500, verbose=True, tol=0.000001,
                                   solver = 'adam', activation = 'relu',
                                   hidden_layer_sizes = (20,20))
neuralNetwork_train.fit(X_treinamento, y_treinamento)
neuralNetwork_previsao = neuralNetwork_train.predict(X_test)

print(accuracy_score(y_test, neuralNetwork_previsao))

