import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import pickle
df = pd.read_csv(r'C:\Users\Tahir Mehmood\Downloads\carprices.csv')
le = LabelEncoder()
dfle = df
dfle['encode'] = le.fit_transform(dfle.CarModel)
print(dfle)
X = dfle
X = X.drop(columns=['CarModel', 'SellPrice'])
print(X)
Y = dfle.SellPrice
print(Y)
model = LinearRegression()
model.fit(X,Y)
print(model.predict([[22500, 2, 1]]))
print(model.score(X,Y))
with open('model_pickle', 'wb') as f:
    pickle.dump(model,f)