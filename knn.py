import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from math import sqrt
from sklearn.metrics import mean_squared_error
from sklearn import tree

df=pd.read_csv(r'D:\thesis\scd.csv')

X = df[["country","year","sex","age","population","gdp_per_capita ($)"]]
y = df[["suicides_no"]]

lb = LabelEncoder()
X["country_code"] = lb.fit_transform(df["country"])
X["sex_code"] = lb.fit_transform(df["sex"])
X["age_code"] = lb.fit_transform(df["age"])

del X["country"]
del X["sex"]
del X["age"]

K = 10 #Number of neighbors

#Train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
x_train_scaled = scaler.fit_transform(X_train)
x_train = pd.DataFrame(x_train_scaled)
x_test_scaled = scaler.fit_transform(X_test)
x_test = pd.DataFrame(x_test_scaled)

from sklearn import neighbors



rmse_val = [] #to store rmse values for different k
model = neighbors.KNeighborsRegressor(n_neighbors = K)
model.fit(x_train, y_train)  #fit the model
pred=model.predict(x_test) #make prediction on test set
error = sqrt(mean_squared_error(y_test,pred)) #calculate rmse
print('RMSE value for k = ' , K , 'is:', error, 'with no cross validation')

print("Below are all cross validation values and the final CV RMSE")
#### Fit DT with cross validation
n_split = 10
kf = KFold(n_splits= n_split, random_state = 10 , shuffle = True)
rmse = 0

for train_index, test_index in kf.split(X,y):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    x_train_scaled = scaler.fit_transform(X_train)
    x_train = pd.DataFrame(x_train_scaled)
    x_test_scaled = scaler.fit_transform(X_test)
    x_test = pd.DataFrame(x_test_scaled)

    model = neighbors.KNeighborsRegressor(n_neighbors = K)
    model.fit(x_train, y_train)  #fit the model
    y_pred = model.predict(x_test) #make prediction on test set
    curr_rmse = sqrt(mean_squared_error(y_test,y_pred)) #calculate rmse
    rmse = rmse +  curr_rmse
    print('RMSE value for k= ' , K , 'is:', curr_rmse)

final_rmse = rmse / n_split
print("cross validation RMSE = " + str(final_rmse))