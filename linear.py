import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv(r'C:\Users\hp\OneDrive\Desktop\ML project\Linear Regression\salary_data.csv')
# print(df.head())
# print(df.tail())
# print(df.isnull().sum())
X= df['YearsExperience'].values.reshape(-1,1)
# print(X)
Y = df['Salary']
# print(Y)
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.33,random_state=2)
# print(X_train)
# print(Y_train)
model = LinearRegression()
lr = model.fit(X_train,Y_train)
Y_pred = model.predict(X_test)
# print(Y_pred)
plt.scatter(X_test,Y_test,color='red',label='Actual')
plt.plot(X_test,Y_pred, color='blue',label='Predicted')
plt.xlabel("Years Exp")
plt.ylabel("Salary")
plt.title("Salarys Vs Years Exp")
plt.show()