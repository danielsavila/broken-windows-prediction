import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

path = "c:/Users/danie/OneDrive/Documents/GitHub/broken-windows-prediction/data"
os.chdir(path)

df = pd.read_csv("monthly_count_df.csv").drop("Unnamed: 0", axis = 1)

# NOTE THAT THIS ANALYSIS IS NOT PREDICTING NEXT MONTHS CRIME, IT IS PREDICTING CURRENT MONTH CRIME
# I.E. PREDICTING JAN CRIME USING JAN POTHOLES + JAN GRAFFITI

pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", None)

seed = 1234567

#xgboost does not take in categorical features, need to make the community column categorical

df["community"] = df["community"].astype("category")

x_train, x_test, y_train, y_test = train_test_split(df.loc[:, "year":"monthly_count_potholes"], 
                                                    df["monthly_count_crime"], 
                                                    train_size = .7, 
                                                    random_state = seed, 
                                                    shuffle = True)

n_est_list = np.logspace(.00001, 10, num = 10, base = 2)
n_est_list = [n.astype(int) for n in n_est_list]
rmselist = []
rsquare_list = []

#also added the "enable_categorical" to accomodate community feature
for estimator in n_est_list:
    xgb = xgboost.XGBRegressor(max_depth = 4, 
                               n_estimators = estimator, 
                               objective = "reg:squarederror", 
                               booster = "gbtree",
                               random_state = seed,
                               enable_categorical = True)
    fit_xgb = xgb.fit(x_train, y_train)
    y_pred = fit_xgb.predict(x_train)
    rmselist.append(np.sqrt(mean_squared_error(y_pred, y_train)))
    rsquare_list.append(fit_xgb.score(x_train,y_train))
    
    plt.scatter(x_train["monthly_count_graffiti"], y_pred, c = "blue", s = 2, label = "xgb +" + estimator.astype(str))
    plt.scatter(x_train["monthly_count_graffiti"], y_train, c = "red", s = 2, label = "true value")
    plt.grid(axis = "both")
    plt.xlabel("monthly count graffiti")
    plt.ylabel("predicted, true values")
    plt.legend()
    plt.show()
    
results = pd.DataFrame({"n estimators": n_est_list,
                        "rmse": rmselist,
                        "r squared": rsquare_list})

results

xgb = xgboost.XGBRegressor(max_depth =  4,
                           n_estimators = 1024,
                           objective = "reg:squarederror",
                           booster = "gbtree",
                           random_state = seed,
                           enable_categorical = True).fit(x_train, y_train)
xgb_pred = xgb.predict(x_test)

rmse = np.sqrt(mean_squared_error(xgb_pred, y_test))
rsquare = xgb.score(x_test, y_test)

print(f"root mean squared error: {round(rmse, 5)}")
print(f"rsquared: {round(rsquare, 5)}")