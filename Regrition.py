# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 23:59:18 2022

@author: Tweety
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory



df = pd.read_csv('diabetes.csv')
df
df.isnull().sum()

for i in df.columns:
  print(df[i].value_counts())
  print()
  
import seaborn as sns
def countplt_fn(x,y,series,xsize,ysize,xtick):
  plt.figure(figsize=(xsize,ysize))
  sns.countplot(series[x], hue=series[y])
  plt.xticks(rotation=xtick)
  plt.show

countplt_fn('Age','Outcome',df,30,6,0)
countplt_fn('BloodPressure','Outcome',df,30,6,0)

countplt_fn('BMI','Outcome',df,70,20,90)
countplt_fn('Glucose','Outcome',df,60,20,90)
countplt_fn('Pregnancies','Outcome',df,30,6,0)
countplt_fn('SkinThickness','Outcome',df,30,6,0)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
from sklearn.metrics import f1_score
x = df.drop(columns='Outcome')
y = df['Outcome']

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.33, random_state = 42)
from sklearn.preprocessing import StandardScaler
st = StandardScaler()
x_train = st.fit_transform(x_train)
x_test = st.fit_transform(x_test)
x_train = pd.DataFrame(x_train)
x_test = pd.DataFrame(x_test)

lgr =LogisticRegression()
rfe = RFE(lgr,7) #7 are sufficient for prediction
rfe.fit(x_train,y_train)
rfe_features = list(x_train.columns[rfe.support_])
rfe_x_train = x_train[rfe_features]

lgr_1 = LogisticRegression()
lgr_1.fit(rfe_x_train,y_train)  

LogisticRegression()

#F1 scores==> {[8 : 0.80938416, 0.61077844], [7: 0.80938416, 0.61077844], [6: 0.80351906, 0.5988024] }
y_pred = lgr_1.predict(x_test[rfe_features])
f1_score_array = f1_score(y_test,y_pred, average=None)
dict_rfe = {"Features": list(rfe_features), "F1 Score":f1_score_array}
dict_rfe

features = []
for i in list(rfe_features):
  #print(df.columns[i])
  features.append(df.columns[i])
X = df[features]
Y = df['Outcome']
#features selected by the model
features

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size = 0.4, random_state = 42)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape

#scaling
X_train = st.fit_transform(X_train)
X_test = st.fit_transform(X_test)
X_train = pd.DataFrame(X_train)
X_test = pd.DataFrame(X_test)

LR = LogisticRegression()
LR.fit(X_train,Y_train)

Y_pred = LR.predict(X_test)
print("Accuracy of Logistic Regression Model is {:.2f}".format(LR.score(X_test,Y_test)))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
print(cm)
inc_pred = (Y_test != Y_pred).sum()
inc_pred

from sklearn.metrics import classification_report
print(classification_report(Y_test,Y_pred))
#reshaping
Y_train = Y_train.values.reshape(-1,1)
Y_test = Y_test.values.reshape(-1,1)

from sklearn.metrics import roc_auc_score, roc_curve
roc_score = roc_auc_score(Y_test,LR.predict(X_test))
fpr, tpr, thr = roc_curve(Y_test, LR.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr,tpr, label='Logistic Regression Area = {:.2f}'.format(roc_score))
plt.plot([0,1],[0,1],'r--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC : Receiver Operating Characteristics')
plt.legend()
plt.show()