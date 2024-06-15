import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import io

df=pd.read_csv('/Crop_recommendation (2) .csv')

df.head(6)

df.tail()

df.isnull().sum()

df.info()

df['label'].value_counts()

x=df.drop('label',axis=1)
y=df['label']

x.info()

y.info()

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=1,test_size=0.2)

x_train.info()

y_train.info()

from sklearn.linear_model import LogisticRegression

model=LogisticRegression()
model.fit(x_train,y_train)

y_pred1=model.predict(x_test)

from sklearn.metrics import accuracy_score
logistic_reg_acc=accuracy_score(y_test,y_pred1)
print("logistic accuracy is" +str(logistic_reg_acc))

from sklearn.tree import DecisionTreeClassifier
model2=DecisionTreeClassifier()

model2.fit(x_train, y_train)

y_pred3=model2.predict(x_test)

decision_acc=accuracy_score(y_test,y_pred3)

print("Decision tree accuracy is "+str(decision_acc))

from sklearn.ensemble import RandomForestClassifier
model3=RandomForestClassifier()
model3.fit(x_train,y_train)
y_pred4=model3.predict(x_test)

ramdom_acc=accuracy_score(y_test,y_pred4)

print("Random forest accuracy is"+str(ramdom_acc))

import joblib

filename='crop app'

joblib.dump(model2,'crop app')

app=joblib.load('crop app')

#user input array:

arr=[[90,42,43,20.879744,82.002744,6.502985,202.935536]]
y_pred5=app.predict(arr)
y_pred5


#code for graphs:
#logistic Regression
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred1,color='black',alpha=0.5)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='blue',linewidth=3)
plt.xlabel('Actual')
plt.ylabel('predicted')
plt.title('Actual vs crop recommendation progression')
plt.grid('True')
plt.show()

#Decision tree:
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred3,color='black',alpha=0.5)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='blue',linewidth=3)
plt.xlabel('Actual')
plt.ylabel('predicted')
plt.title('Actual vs crop recommendation progression')
plt.grid('True')
plt.show()

#Random forest:
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred4,color='black',alpha=0.5)
plt.plot([y_test.min(),y_test.max()],[y_test.min(),y_test.max()],color='blue',linewidth=3)
plt.xlabel('Actual')
plt.ylabel('predicted')
plt.title('Actual vs crop recommendation progression')
plt.grid('True')
plt.show()

