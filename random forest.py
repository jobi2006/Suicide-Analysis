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

#Train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=100)
model = regr.fit(X, y)

### Random FOrest
print(regr.feature_importances_)
y_pred = regr.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE =")
print(rmse)

#### Fit RF with cross validation
n_split = 10
kf = KFold(n_splits= n_split, random_state = 10 , shuffle = True)
rmse = 0
for train_index, test_index in kf.split(X,y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(X_train.shape)
    print(y_train.shape )
    RF_model = regr.fit(X_train, y_train)
    y_pred = RF_model.predict(X_test)
    rmse = rmse +  sqrt(mean_squared_error(y_test, y_pred))
final_rmse = rmse / n_split
print("cross validation RMSE =" + str(final_rmse))
regr = RandomForestRegressor(max_depth=3, random_state=0, n_estimators=100)
model = regr.fit(X, y)

### Random FOrest
print(regr.feature_importances_)
y_pred = regr.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE =")
print(rmse)

#### Fit RF with cross validation
n_split = 3
kf = KFold(n_splits= n_split)
rmse = 0
for train_index, test_index in kf.split(X,y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print(X_train.shape)
    print(y_train.shape )
    RF_model = regr.fit(X_train, y_train)
    y_pred = RF_model.predict(X_test)
    rmse = rmse +  sqrt(mean_squared_error(y_test, y_pred))
final_rmse = rmse / n_split
print("cross validation RMSE =" + str(final_rmse))