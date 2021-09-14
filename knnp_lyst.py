import numpy as np
import pandas as pd

url="https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

names=['sepal-length','sepal-width','petal-length','petal-width','Class']

dataset=pd.read_csv(url,names=names)
#print(dataset)

x= dataset.iloc[:,:-1].values
#print(x)
y=dataset.iloc[:,4].values
#print(y)

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.20)

from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit(x_train)
x_train=scalar.transform(x_train)
x_test=scalar.transform(x_test)
print(x_train)
print(x_test)

from sklearn.neighbors import KNeighborsClassifier
classifier= KNeighborsClassifier(n_neighbors=12)
classifier.fit(x_train,y_train)
y_pred=classifier.predict(x_test)
print(y_pred)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,y_pred))



