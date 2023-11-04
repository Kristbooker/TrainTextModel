import joblib
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

le = LabelEncoder()

train_text = joblib.load("train_text_vector3.pkl")
test_text = joblib.load("test_text_vector3.pkl")

df_train = pd.read_csv("/Users/krist/TrainModel/emotionDataset/train.txt", sep=";", names=["Text", "Emotion"])
df_test = pd.read_csv("/Users/krist/TrainModel/emotionDataset/test.txt", sep=";", names=["Text", "Emotion"])



X_train = train_text
y_train = le.fit_transform(df_train['Emotion'])

X_test = test_text
y_test = le.fit_transform(df_test['Emotion'])

# print(y_test.tolist())
clf = RandomForestClassifier(n_estimators=10 , random_state=42)
clf.fit(X_train, y_train)

# Evaluate on the test set
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

write_path_model="TextModel2.pkl"
pickle.dump(clf,open(write_path_model,"wb"))
print("done")