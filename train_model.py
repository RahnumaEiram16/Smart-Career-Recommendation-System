import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

df = pd.read_csv("career_dataset.csv")

X = df.drop("Career", axis=1)

le = LabelEncoder()
X["Interest"] = le.fit_transform(X["Interest"])

y = df["Career"]

model = DecisionTreeClassifier()
model.fit(X, y)

pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))

print("Model Trained Successfully!")