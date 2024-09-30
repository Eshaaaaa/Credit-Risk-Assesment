import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score

import os
for dirname, _, filenames in os.walk('D:\3 rd yr\Mini_Project'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

df = pd.read_csv('D:/3 rd yr/Mini_Project/credit/bankloans.csv')
df.head()
df.isnull().sum()
df.value_counts()
df = df.dropna()
print(df)
fig,ax = plt.subplots(figsize=(5,5))
sns.lineplot(x='age',y='income',data=df,ax=ax)

fig,ax = plt.subplots(figsize=(5,5))
sns.lineplot(x='age',y='debtinc',data=df,ax=ax)

df['default'].value_counts()

x=df.drop(['default'],axis=1)
y=df['default']
print(x)
print(y)

# Split the data into training and testing sets
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42)

sc = StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(xtrain,ytrain)

rfc.score(xtest,ytest)

rfc2 = cross_val_score(estimator=rfc,X=xtrain,y=ytrain,cv=10)
rfc2.mean()

#SVM
sv = SVC()
sv.fit(xtrain,ytrain)
sv.score(xtest,ytest)

model = GridSearchCV(sv,{
    'C':[0.1,0.2,0.4,0.8,1.2,1.8,4.0,7.0],
    'gamma':[0.1,0.4,0.8,1.0,2.0,3.0],
    'kernel':['rbf','linear']
},scoring='accuracy',cv=10)

model.fit(xtrain,ytrain)

model.best_params_

model2 = SVC(C=0.1,gamma=0.1,kernel='linear')
model2.fit(xtrain,ytrain)
model2.score(xtest,ytest)

lr = LogisticRegression()
lr.fit(xtrain,ytrain)
lr.score(xtest,ytest)

yp = lr.predict(xtest)
c= confusion_matrix(ytest,yp)
fig ,ax = plt.subplots(figsize=(10,5))
sns.heatmap(c,ax=ax)

model_accuracies = {
    'Random Forest': rfc_accuracy,
    'SVC (Best)': svc_best_accuracy,
    'Logistic Regression': lr_accuracy
}

best_model_name = max(model_accuracies, key=model_accuracies.get)
print(f"The best model is {best_model_name} with an accuracy of {model_accuracies[best_model_name]:.2f}")
