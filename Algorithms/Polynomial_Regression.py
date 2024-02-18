import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

dataset = pd.read_csv("../data/Position_Salaries.csv")
#Feature set
X = dataset.iloc[:,1:-1].values
#Dependable variable
y = dataset.iloc[:,-1].values

lin_regressor = LinearRegression()
lin_regressor.fit(X,y)

poly_reg = PolynomialFeatures(degree=5)
X_poly = poly_reg.fit_transform(X)

lin_regressor_poly = LinearRegression()
lin_regressor_poly.fit(X_poly,y)

plt.scatter(X,y,color='red')
plt.plot(X,lin_regressor.predict(X),color='blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


plt.scatter(X,y,color='red')
plt.plot(X,lin_regressor_poly.predict(X_poly),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

X_grid = np.arange(min(X),max(X),0.1)
print(X_grid)
X_grid = X_grid.reshape(len(X_grid),1)
print(X_grid)
plt.scatter(X,y,color='red')
plt.plot(X_grid,lin_regressor_poly.predict(poly_reg.fit_transform(X_grid)),color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

print(lin_regressor.predict([[6.5]]))
print(lin_regressor_poly.predict(poly_reg.fit_transform([[6.5]])))