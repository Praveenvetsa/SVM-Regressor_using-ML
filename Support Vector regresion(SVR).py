#  Support Vector Regression

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv(r'C:\Users\LENOVO\OneDrive\Desktop\fsds materials\fsds\3. Aug\21st\EMP SAL.csv')

X = dataset.iloc[:,1:2].values # Independent variables
y = dataset.iloc[:,2].values # Dependent variables

# Fitting SVR to the dataset
'''from sklearn.svm import SVR
regressor = SVR()
regressor.fit(X,y)

# Prediciting a new result
y_pred = regressor.predict([[6.5]])
y_pred'''

'''from sklearn.svm import SVR
regressor_poly = SVR(kernel = 'poly', degree = 6) 
# degree 3,4,5,6
regressor_poly.fit(X,y)

y_pred1 = regressor_poly.predict([[6.5]])
y_pred1'''

'''from sklearn.svm import SVR
regressor = SVR(kernel = 'sigmoid',degree = 6)
# degree = 3,4,5,6
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])
y_pred'''

'''from sklearn.svm import SVR
reg  =SVR(kernel ='linear')
reg.fit(X,y)
y_pred = reg.predict([[6.5]])'''

from sklearn.svm import SVR
regressor = SVR(kernel ='poly', degree =7,)
regressor.fit(X,y)

y_pred = regressor.predict([[6.5]])
y_pred

#Visuvalizing the SVR results
%matplotlib inline
plt.scatter(X,y, color = 'red')
plt.plot(X,regressor.predict(X),color ='blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualising the SVR results (for higher resolution and smoother curve)
X_grid=np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color = 'red')
plt.plot(X_grid, regressor.predict(X_grid),color = 'blue')
plt.title('Truth or Bluff (SVR)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()







