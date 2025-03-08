### NAME: SURYA P <br>
### REG NO: 212224230280

# IMPLEMENTATION OF LINEAR REGRESSION  MODEL USING GRADIENT DESCENT

## AIM :

To write a program to predict the profit of a city using the linear regression model with gradient descent.

## EQUIPMENTS REQUIRED :

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## ALGORITHM :

1.Import the required library and read the dataframe.

2.Write a function computeCost to generate the cost function.

3.Perform iterations og gradient steps with learning rate.

4.Plot the Cost function using Gradient Descent and generate the required graph.

## PROGRAM :

```
# Program to Implement Linear Regression Using Gradient Descent

# Developed by: Surya P
# RegisterNumber: 212224230280


import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
def linear_regression(x1,y,learning_rate=0.01,num_iters=1000):
  X = np.c_[np.ones(len(X1)),X1]
  theta = np.zeros(X.shape[1]).reshape(-1,1)
  for _ in range(num_iters):
    predictions = (X).dot(theta).reshape(-1,1)
    errors=(predictions - y ).reshape(-1,1)
    theta -= learning_rate*(1/len(X1))*X.T.dot(errors)
  return theta
data=pd.read_csv("50_Startups.csv",header=None)
print(data.head)
X=(data.iloc[1:,:-2].values)
print(X)
X1=X.astype(float)
scaler=StandardScaler()
y=(data.iloc[1:,-1].values).reshape(-1,1)
print(y)
X1_Scaled=scaler.fit_transform(X1)
Y1_Scaled=scaler.fit_transform(y)
print(X1_Scaled)
print(Y1_Scaled)
theta= linear_regression(X1_Scaled,Y1_Scaled)
new_data=np.array([165349.2,136897.8,471784.1]).reshape(-1,1)
new_Scaled=scaler.fit_transform(new_data)
prediction=np.dot(np.append(1,new_Scaled),theta)
prediction=prediction.reshape(-1,1)
pre=scaler.inverse_transform(prediction)
print(prediction)
print(f"Predicted value: {pre}")
```

## OUTPUT :

![Screenshot 2025-03-08 091432](https://github.com/user-attachments/assets/d24b3162-02aa-490d-a8db-70d6e0d792ec)

![image](https://github.com/user-attachments/assets/49152612-50e6-41fb-ab73-d71cfa4d3f2b)

![image](https://github.com/user-attachments/assets/c09dcf5f-1c62-4309-8539-17f3634ab9f0)

![image](https://github.com/user-attachments/assets/41ef0e94-6d3a-4fad-bac4-c9145e99d482)

![image](https://github.com/user-attachments/assets/07db7da5-b8db-4d79-8e6a-640f66f94445)

![image](https://github.com/user-attachments/assets/1ac6afb1-3da6-452a-acad-f60fea32e743)

![image](https://github.com/user-attachments/assets/7d35aa7a-b623-4ee7-9607-78f1a23ae97b)

## RESULT :

Thus the program to implement the linear regression using gradient descent is written and verified using python programming.
