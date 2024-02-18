import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv("../data/Salary_Data.csv")
#Feature set
X = dataset.iloc[:,:-1].values
#Dependable variable
y = dataset.iloc[:,-1].values

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=1)

regressor = LinearRegression()
regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

print(regressor.predict([[12]]))

plt.scatter(X_train, y_train, color='red')
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title("Salary vs experience for training set")
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()

plt.scatter(X_test, y_test, color='red')
plt.plot(X_train,regressor.predict(X_train), color="blue")
plt.title("Salary vs experience for test set")
plt.xlabel('Years of experience')
plt.ylabel('salary')
plt.show()