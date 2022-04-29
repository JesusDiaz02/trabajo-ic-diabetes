import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from warnings import simplefilter
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import neighbors

simplefilter(action='ignore', category=FutureWarning)


url ='diabetes.csv'
data = pd.read_csv(url)

data.drop(['Glucose', 'Insulin', 'BMI', 'DiabetesPedigreeFunction'],axis=1, inplace=True)
data.Age.replace(np.nan,34, inplace=True)
rangos = [0, 8, 15, 18, 25, 40, 60,100]
nombres = ['1','2','3','4','5','6','7']
data.Age=pd.cut(data.Age, rangos, labels=nombres)
data.dropna(axis=0,how='any',inplace=True)

#partir la tabla en dos
data_train = data[:391]
data_test = data[391:]

x=np.array(data_train.drop(['Outcome'], 1))
y=np.array(data_train.Outcome)# 0 sale 1 no sale

x_train, x_test, y_train, y_test= train_test_split(x, y, test_size=0.2)

x_test_out = np.array(data_test.drop(['Outcome'], 1))
y_test_out = np.array(data_test.Outcome)# 0 sale 1 no sale