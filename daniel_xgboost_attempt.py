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
df = df.drop("previous_month", axis = 1)

# NOTE THAT THIS ANALYSIS IS NOT PREDICTING NEXT MONTHS CRIME, IT IS PREDICTING CURRENT MONTH CRIME
# I.E. PREDICTING JAN CRIME USING JAN POTHOLES + JAN GRAFFITI

pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", None)

seed = 1234567

#xgboost does not take in categorical features, need to make the community column categorical

df["community"] = df["community"].astype("category")

x_train, x_test, y_train, y_test = train_test_split(df.loc[:, "community": "month_previous"], 
                                                    df["realized_crime"], 
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
    
    plt.scatter(x_train["previous_month_potholes"], y_pred, c = "blue", s = 2, label = "xgb +" + estimator.astype(str))
    plt.scatter(x_train["previous_month_potholes"], y_train, c = "red", s = 2, label = "true value")
    plt.grid(axis = "both")
    plt.xlabel("previous month's potholes")
    plt.ylabel("crimes in a given month")
    plt.legend()
    plt.show()
    
results_train = pd.DataFrame({"n estimators": n_est_list,
                        "rmse, train": rmselist,
                        "r squared, train": rsquare_list})


#instantiating final model
xgb = xgboost.XGBRegressor(max_depth =  4,
                           n_estimators = 1024,
                           objective = "reg:squarederror",
                           booster = "gbtree",
                           random_state = seed,
                           enable_categorical = True).fit(x_train, y_train)
xgb_pred = round(pd.Series(xgb.predict(x_test)), 2)

rmse = np.sqrt(mean_squared_error(xgb_pred, y_test))
rsquare = xgb.score(x_test, y_test)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("training set results:")
display(results_train)

print("~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("test set results:")
print(f"root mean squared error test: {round(rmse, 5)}")
print(f"rsquared test: {round(rsquare, 5)}")

predictions = pd.DataFrame({"realized_crime": y_test, 
                            "predicted_crime": round(xgb_pred, 2)}).dropna().reset_index()
# the first month in the dataset will not be able to generate realized crime values
# and the last month in the dataset will not be able to generate predicted crime values
# we see this in the predictions dataframe as NaN values, therefore we filter these out.

predictions.head(10)