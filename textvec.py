import pandas as pd
import joblib
import requests  
import numpy as np

df_train = pd.read_csv("/Users/krist/TrainModel/emotionDataset/train.txt", sep=";", names=["Text", "Emotion"])
df_test = pd.read_csv("/Users/krist/TrainModel/emotionDataset/test.txt", sep=";", names=["Text", "Emotion"])

#joy sadness anger fear love surprise


def text2vec(txt):
    url = "http://localhost:8080/api/genvec"

    response = requests.get(url, json={"text_data":txt})

    if response.status_code == 200:
        try:
            return response.json()
        except requests.exceptions.JSONDecodeError as e:
            print("JSON DECODE ERROR:",e)
    else :
        print("API request error. status code:", response.status_code)
        return None
    

def text_to_vectors(df):
    vectors = []
    for text in df["Text"]:
        vec = text2vec(text)
        if vec:
            vectors.append(vec["VECTOR"][0])
        
    return vectors

train_vectors = text_to_vectors(df_train)
test_vectors = text_to_vectors(df_test)



joblib.dump(train_vectors,'train_text_vector3.pkl')
joblib.dump(test_vectors,'test_text_vector3.pkl')