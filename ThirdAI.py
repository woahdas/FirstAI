#This is the best one so far.
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('/kaggle/input/titanic/train.csv')
test = pd.read_csv('/kaggle/input/titanic/test.csv')


train = train.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
test = test.drop(['Name','SibSp','Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],axis=1)
for df in [train,test]:
        df['Sex_binary']=df['Sex'].map({'male':1,'female':0})
    

train['Age'] = train['Age'].fillna(0)
test['Age'] = test['Age'].fillna(0)

features = ['Pclass','Age','Sex_binary']
target = 'Survived'

###
###

logreg = LogisticRegression()

logreg.fit(train[features],train[target])

logreg_predictions = logreg.predict(test[features])
#logreg_data = pd.read_csv("/kaggle/input/titanic/test.csv")

submission = pd.DataFrame({'PassengerId':test['PassengerId'],'Survived':logreg_predictions})
submission.to_csv('submission2.csv', index=False)
