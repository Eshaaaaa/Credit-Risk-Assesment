import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

# Load the dataset
df = pd.read_csv('../input/credit-risk-analysis-for-extending-bank-loans/bankloans.csv')

# Display the first few rows of the dataset
df.head()

# Check for missing values
df.isnull().sum()

# Drop rows with missing values
df = df.dropna()

# Visualize the relationship between 'age' and 'income'
fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(x='age', y='income', data=df, ax=ax)
plt.show()

# Visualize the relationship between 'age' and 'debtinc' (debt-to-income ratio)
fig, ax = plt.subplots(figsize=(20, 10))
sns.lineplot(x='age', y='debtinc', data=df, ax=ax)
plt.show()

# Display the distribution of the target variable ('default')
df['default'].value_counts()

# Define features (X) and target (y)
X = df.drop(['default'], axis=1)
y = df['default']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train and evaluate a Random Forest classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
print(f'Random Forest Accuracy: {rfc.score(X_test, y_test):.2f}')

# Perform cross-validation on Random Forest
rfc_cv_scores = cross_val_score(rfc, X_train, y_train, cv=10)
print(f'Random Forest Cross-validation Accuracy: {rfc_cv_scores.mean():.2f}')

# Train and evaluate a Support Vector Classifier (SVC)
svc = SVC()
svc.fit(X_train, y_train)
print(f'SVC Accuracy: {svc.score(X_test, y_test):.2f}')

# Hyperparameter tuning for SVC using GridSearchCV
param_grid = {'C': [0.1, 0.2, 0.4, 0.8, 1.2, 1.8, 4.0, 7.0],
              'gamma': [0.1, 0.4, 0.8, 1.0, 2.0, 3.0],
              'kernel': ['rbf', 'linear']}
grid_search = GridSearchCV(SVC(), param_grid, scoring='accuracy', cv=10)
grid_search.fit(X_train, y_train)
print(f'Best Hyperparameters for SVC: {grid_search.best_params_}')

# Train and evaluate the best SVC model
svc_best = SVC(C=0.1, gamma=0.1, kernel='linear')
svc_best.fit(X_train, y_train)
print(f'Best SVC Model Accuracy: {svc_best.score(X_test, y_test):.2f}')

# Train and evaluate a Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
print(f'Logistic Regression Accuracy: {lr.score(X_test, y_test):.2f}')

# Generate predictions and evaluate the Logistic Regression model using a confusion matrix
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix using Seaborn
fig, ax = plt.subplots(figsize=(20, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
