import tensorflow as tf
import keras
import numpy as np
import sklearn
import pandas as pd
from sklearn import linear_model
from sklearn.utils import shuffle

# -------------Linear Regression----------------
data = pd.read_csv("student-mat.csv", sep=";")
# print(data.head())

# data set of 5 different attributes
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# G3 is the Final Grade of Students.
# based on all attributes in data set
# we will predict G3
predict = "G3"

# We're taking data in X from above dataSet
# But we've to drop 'G3', BCOZ we'll predict G3
# using attributes in above dataSet
# so X will return new data set that doesn't have G3
X = np.array(data.drop([predict], 1))

# X is our training data, Based on
# X we'll predict Y
Y = np.array(data[predict])

x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(X, Y, test_size=0.01)

# drawing linear-fit line Y = mX + b
linear = linear_model.LinearRegression()
linear.fit(x_train, y_train)
accuracy = linear.score(x_test, y_test)

# Y = mX + b, Y is the predicted accuracy
print("y = ", accuracy)

# coefficient of five different variables
print("Coefficients (m) are: \n", linear.coef_)
# y= mx+b, here b is the intercept point on Y-axis
print("Intercept(b): \n", linear.intercept_)

predictions = linear.predict(x_test)

for x in range(len(predictions)):
    print("Prediction: ", predictions[x], "Learn-X: ", x_test[x], "Learn-Y: ", y_test[x])





# -------------- END-OF-LINEAR-REGRASSION ------------------