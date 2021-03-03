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


#2ND AI, WITH SOME HELP FROM A TUTORIAL
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import sklearn
import sklearn.model_selection
import sklearn.metrics
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.tree
import seaborn as sns
import sklearn.preprocessing
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelBinarizer
from sklearn.linear_model import LogisticRegression

train_pd = pd.read_csv("/kaggle/input/titanic/train.csv")
test_pd = pd.read_csv("/kaggle/input/titanic/test.csv")
data = [train_pd, test_pd]

x = pd.get_dummies(train_pd[features])
y = train_pd["Survived"]
x_test = pd.get_dummies(test_data[features])

for dataset in data:
    mean = train_pd["Age"].mean()
    standard = test_pd["Age"].std()
    isnull = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - standard, mean + standard, size = isnull)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_pd["Age"].astype(int)
    
    

embarked_mode = train_pd['Embarked'].mode()
data = [train_pd, test_pd]
for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].fillna(embarked_mode)
    
data = [train_pd, test_pd]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'travelled_alone'] = 'No'
    dataset.loc[dataset['relatives'] == 0, 'travelled_alone'] = 'Yes'
    
trainnumfeat = list(train_pd.select_dtypes(include=['int64', 'float64', 'int32']).columns)
s_scaler = StandardScaler()
train_pd_s = pd.DataFrame(data = train_pd)
train_pd_s[trainnumfeat] = s_scaler.fit_transform(train_pd_s[trainnumfeat])

encode_col_list = list(train_pd.select_dtypes(include=['object']).columns)
for i in encode_col_list:
    train_pd_s = pd.concat([train_pd_s,pd.get_dummies(train_pd_s[i], prefix=i)],axis=1)
    train_pd_s.drop(i, axis = 1, inplace=True)
    

logreg = LogisticRegression()

logreg.fit(x, y)

logreg_predictions = logreg.predict(x_test)
logreg_data = pd.read_csv("/kaggle/input/titanic/test.csv")
logreg_data.insert((logreg_data.shape[1]),'Survived',logreg_predictions)
logreg_data.to_csv('Titanic1.csv', index=False)




#model.fit(x_train, y)

history = model.fit(x_test,
                    epochs=1,
                    batch_size=128,
                    validation_data=(x_test.empty))

predictions = model.predict(x_test) 

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv('my_submission.csv', index=False)

