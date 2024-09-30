# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pickle

# Load the dataset
data = pd.read_csv('data/bankloans.csv')

# Check for missing values and drop rows with missing data
data = data.dropna()

# Define features (X) and target variable (y)
X = data.drop('default', axis=1)
y = data['default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Logistic Regression model
lr = LogisticRegression()
lr.fit(X_train, y_train)
lr_acc = lr.score(X_test, y_test)
print(f'Logistic Regression Accuracy: {lr_acc * 100:.2f}%')

# Train Random Forest model
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)
rfc_acc = rfc.score(X_test, y_test)
print(f'Random Forest Accuracy: {rfc_acc * 100:.2f}%')

# Train Support Vector Classifier (SVC) model
svc = SVC()
svc.fit(X_train, y_train)
svc_acc = svc.score(X_test, y_test)
print(f'Support Vector Classifier Accuracy: {svc_acc * 100:.2f}%')

# Evaluate the best model using confusion matrix (Logistic Regression in this case)
y_pred = lr.predict(X_test)
cm = confusion_matrix(y_test, y_pred)

# Visualize the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix - Logistic Regression')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

# Save the best model (Logistic Regression) to a file
with open('models/logistic_model.pkl', 'wb') as model_file:
    pickle.dump(lr, model_file)

# Summary of model accuracies
print(f'\nModel Accuracies:')
print(f'Logistic Regression: {lr_acc * 100:.2f}%')
print(f'Random Forest: {rfc_acc * 100:.2f}%')
print(f'Support Vector Classifier: {svc_acc * 100:.2f}%')

