import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score , classification_report
import joblib

# load the dataset 
df=pd.read_csv("pima-indians-diabetes.csv")

#print(df)

# separate the variable
X=df.drop(columns="Outcome")
y=df["Outcome"]

# train the dataset
X_train ,X_test ,y_train ,y_test =train_test_split(X ,y,test_size=0.3 ,random_state=43)

# model
model=LogisticRegression()

model.fit(X_train ,y_train)
y_pred=model.predict(X_test)


report=classification_report(y_pred ,y_test)
#print(report)


accuracy=accuracy_score(y_pred ,y_test)
#print(accuracy)

# export the model as pkl file

joblib.dump(model ,"log_reg.pkl")