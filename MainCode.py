import os
import keras
import sklearn.ensemble
from keras import layers
from keras import models
from tensorflow.python.keras.models import Sequential
import pandas as pd
from keras.optimizers import RMSprop


train_data = pd.read_csv("/kaggle/input/titanic/train.csv")
test_data = pd.read_csv("/kaggle/input/titanic/test.csv")


y = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
x_train = pd.get_dummies(train_data[features])
x_test = pd.get_dummies(test_data[features])

#model.add(layers.Embedding(features, 128, input_length=max))

model = Sequential()

model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.MaxPool1D(5))
model.add(layers.Conv1D(32, 7, activation='relu'))
model.add(layers.GlobalMaxPool1D())
model.add(layers.Dense(1))


model.compile(optimizer=RMSprop(lr=1e-4),
              loss='binary_crossentropy',
              metrics=['acc'])

#model.fit(x_train, y)

history = model.fit(x_test,
                    epochs=1,
                    batch_size=128,
                    validation_data=(x_test.empty))

predictions = model.predict(x_test) 

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)
print("Your submission was successfully saved!")
