import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

Codtech - Riya Singh - CITS9175

Load your dataset: df = pd.read_csv('creditcard.csv')
X = [[1,100,0],[2,200,1],[1,50,0],[5,1000,1]]*100
y = [0,1,0,1]*100

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)

print("Accuracy:", accuracy_score(y_test, model.predict(X_test)))
