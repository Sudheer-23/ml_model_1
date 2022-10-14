import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.read_csv('IceCreamData.csv')
df.info()
df.head()

X = df['Temperature']
Y = df['Revenue']

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.3)
print(len(X_train))
print(len(X_test))
print(len(Y_train))
print(len(Y_test))
X_train.ndim

model = LinearRegression()

X_train = np.array([X_train])
X_train.ndim

model.fit(X_train.reshape(-1,1), Y_train)

X_test = np.array([X_test])
X_test.ndim

Predict = model.predict(X_test.reshape(-1 ,1))
Predict

Y_test

r2_score(Y_test, Predict)

print(model.predict([[45]]))

pickle.dump(model,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))