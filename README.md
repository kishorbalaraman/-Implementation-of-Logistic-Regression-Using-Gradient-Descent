# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries and create the dataset for input and output values.
2. Initialize parameters and define the sigmoid function for Logistic Regression.
3. Train the model using Gradient Descent by updating weights and bias repeatedly.
4. Predict the output and evaluate the model accuracy. 

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: KISHOR B
RegisterNumber:  212225230141
*/

# Step 1: Import Libraries
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# Step 2: Create Dataset
data = {
    'Hours': [1, 2, 3, 4, 5, 6, 7, 8],
    'Pass':  [0, 0, 0, 0, 1, 1, 1, 1]
}

df = pd.DataFrame(data)

print("Dataset:")
print(df)

# Step 3: Select Input and Output
X = df['Hours'].values
y = df['Pass'].values

# Step 4: Normalize Input
X = X / np.max(X)

# Step 5: Initialize Parameters
m = 0
b = 0
learning_rate = 0.1
epochs = 1000

# Step 6: Sigmoid Function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Step 7: Gradient Descent Training
n = len(X)

for i in range(epochs):

    # Linear Equation
    z = m * X + b

    # Sigmoid Prediction
    y_pred = sigmoid(z)

    # Calculate Gradients
    dm = (1/n) * np.sum((y_pred - y) * X)
    db = (1/n) * np.sum(y_pred - y)

    # Update Parameters
    m = m - learning_rate * dm
    b = b - learning_rate * db

# Step 8: Final Predictions
z = m * X + b
predictions = sigmoid(z)

final_predictions = [1 if i >= 0.5 else 0 for i in predictions]

# Step 9: Accuracy
accuracy = accuracy_score(y, final_predictions)

print("\nPredicted Output:")
print(final_predictions)

print("\nAccuracy:")
print(accuracy)

# Step 10: Confusion Matrix
cm = confusion_matrix(y, final_predictions)

print("\nConfusion Matrix:")
print(cm)

# Step 11: Classification Report
report = classification_report(y, final_predictions)

print("\nClassification Report:")
print(report)

# Step 12: Predict New Value
hours = float(input("\nEnter Study Hours: "))

hours = hours / 8

result = sigmoid(m * hours + b)

if result >= 0.5:
    print("Student Will Pass")
else:
    print("Student Will Fail")

```

## Output:
<img width="519" height="541" alt="image" src="https://github.com/user-attachments/assets/4e41f494-e1c5-4604-a83e-a6d8b5f091b7" />


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

