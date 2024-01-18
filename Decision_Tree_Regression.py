import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv("../data/Position_Salaries.csv")
#Feature set
X = dataset.iloc[:,1:-1].values
#Dependable variable
y = dataset.iloc[:,-1].values

regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)
print(regressor.predict([[6.5]]))

X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Polybomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()