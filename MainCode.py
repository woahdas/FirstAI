import os
import keras
import sklearn.ensemble
from keras import layers
from keras import models
from tensorflow.python.keras.models import Sequential
import pandas as pd

train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
x_train = pd.get_dummies(train_data[features])
x_test = pd.get_dummies(test_data[features])

model = Sequential()
model.add(layers.Embedding(features, 128, input_length=max))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))

model.fit(x_train, y)
predictions = model.predict(x_test)
