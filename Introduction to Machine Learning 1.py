import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


data = pd.read_csv('C:\\Users\\user\\Downloads\\assignments\\Module 6\\data (1).csv')
data.head()
data.shape
x = data.iloc[:, :-1].values
y = data.iloc[:, 1].values

data


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.30, random_state=0)

x_train
y_train

x_test
y_test



regressor = LinearRegression()
regressor.fit(x_train, y_train)

TestPreditions=regressor.predict(x_test)
TestPreditions

TrainPreditions = regressor.predict(x_train)
TrainPreditions


plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.title("Salary vs Experience of Train Dataset")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


plt.scatter(x_test, y_test, color = "blue")
plt.plot(x_test, regressor.predict(x_test), color = "red")
plt.title("Salary vs Experience Test Dataset)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


print(regressor.get_params)

print(regressor.intercept_)

print(regressor.coef_)


r2_score(y_test, TestPreditions)

from sklearn import metrics 
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, TestPreditions))  
print('Mean Squared Error:', metrics.mean_squared_error(y_test, TestPreditions))  
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, TestPreditions)))

