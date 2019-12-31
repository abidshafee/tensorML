import tensorflow as tf
import keras
import numpy as np
import sklearn
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

predict = "G2"

X = np.array(data.drop([predict], 1))
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.01)

linear = linear_model.LinearRegression()

linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

# x = mx + b, x is the predicted accuracy
print("x = ", accuracy)

# coefficient of five different variables
print("Coefficients (m) are: \n", linear.coef_)
# y= mx+b, here b is the intercept point on Y-axis
print("Intercept(b): \n", linear.intercept_)
