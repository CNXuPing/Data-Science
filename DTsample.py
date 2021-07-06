import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

ds=pd.read_csv('D:/JupyterProject/DT/dataset.csv') # dataset
ds.head()

ds['Result']=ds['Result'].map({'Y':1,'N':0})
header=['Age','Income','LoanAmount'] # Missing Value Imputation

ds = ds.fillna(value=ds[header].mean())

ds.head()

X=ds.drop(columns=['Name','Result']).values
Y=ds['Result'].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 40)

print('Shape of X_train=>',X_train.shape)
print('Shape of X_test=>',X_test.shape)
print('Shape of Y_train=>',Y_train.shape)
print('Shape of Y_test=>',Y_test.shape)

# Decision Tree
dtc = DecisionTreeClassifier(criterion = 'entropy', random_state = 42) 
dtc.fit(X_train, Y_train)

# Evaluation on Training set
dtc_pred_train = dtc.predict(X_train)
print('Training: ',f1_score(Y_train,dtc_pred_train))

# Evaluating on Test set
dtc_pred_test = dtc.predict(X_test)
print('Testing: ',f1_score(Y_test,dtc_pred_test))

# Random Forest 
rfc = RandomForestClassifier(criterion = 'entropy', random_state = 45)
rfc.fit(X_train, Y_train)

# Evaluating on Training set
rfc_pred_train = rfc.predict(X_train)
print('Training: ',f1_score(Y_train,rfc_pred_train))

# Evaluating on Test set
rfc_pred_test = rfc.predict(X_test)
print('Testing: ',f1_score(Y_test,rfc_pred_test))
