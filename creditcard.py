import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


dataset = pd.read_csv('creditcard.csv')

print(dataset.shape)  # Check number of rows and columns
print(dataset.head())  # Display first few rows
print(dataset.info())  # Get dataset information
print(dataset.describe())  # Statistical summary of the dataset

print(dataset.isnull().sum())  # Check for any missing values in the dataset

print(dataset['Class'].value_counts())  # Check number of fraudulent and legitimate transactions

# Scale 'Amount' column (since 'Time' is not scaled in typical scenarios)
scaler = StandardScaler()
dataset['NormalizedAmount'] = scaler.fit_transform(dataset['Amount'].values.reshape(-1, 1))
dataset = dataset.drop(['Time', 'Amount'], axis=1)

X = dataset.drop(['Class'], axis=1)
Y = dataset['Class']


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)


model = LogisticRegression(max_iter=1000)
model.fit(X_train, Y_train)


Y_pred = model.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(Y_test, Y_pred)
print("Accuracy:", accuracy)

# Confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(Y_test, Y_pred)
print(cm)

# Classification report
print("\nClassification Report:")
print(classification_report(Y_test, Y_pred))

# Heatmap of confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='d', annot_kws={"size": 16})
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()
