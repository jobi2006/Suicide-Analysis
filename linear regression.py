import pandas as pd
df=pd.read_csv(r'D:\thesis\scd.csv')

X = df[["country","year","sex","age","population","gdp_per_capita ($)"]]
y = df[["suicides_no"]]

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X["country_code"] = lb.fit_transform(df["country"])
X["sex_code"] = lb.fit_transform(df["sex"])
X["age_code"] = lb.fit_transform(df["age"])

del X["country"]
del X["sex"]
del X["age"]
from sklearn.model_selection import train_test_split
#Train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#fit linear regression

lm = linear_model.LinearRegression()
model = lm.fit(X_train, y_train)
y_pred = lm.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE =")
print(rmse)

#### Fit linear regression with cross validation
from sklearn.model_selection import KFold


kf = KFold(n_splits=10)
for train_index, test_index in kf.split(X,y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    lr_model = lm.fit(X_train, y_train)
    y_pred = lr_model.predict(X_test)
    rmse = sqrt(mean_squared_error(y_test, y_pred))
    print("RMSE =")
    print(rmse)