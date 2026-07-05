import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib


df= pd.read_csv("data/loan_data.csv")

X= df.drop(columns=["Approved"])
Y= df["Approved"]

X_train,X_test,Y_train,Y_test= train_test_split(X,Y,test_size=0.3,random_state=42)
model= DecisionTreeClassifier(random_state=42)
model.fit(X_train,Y_train)
Y_pred= model.predict(X_test)
print("Accuracy of the model: ",accuracy_score(Y_test,Y_pred))
joblib.dump(model, "loan_model.pkl")

print("Model saved successfully as loan_model.pkl")