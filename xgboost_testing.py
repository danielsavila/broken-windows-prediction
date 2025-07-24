import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from databricks import sql
import pickle

pd.set_option("display.max_columns", None)
pd.set_option("max_colwidth", None)

seed = 1234567

#loading in data from databricks
with sql.connect(server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME"),
                 http_path       = os.getenv("DATABRICKS_HTTP_PATH"),
                 access_token    = os.getenv("DATABRICKS_TOKEN")) as connection:

  with connection.cursor() as cursor:
        cursor.execute(f"SELECT * FROM workspace.broken_windows_final_data.monthly_count_df")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]

df = pd.DataFrame(rows, columns=columns)

# This analysis is predicting next months crime, using previous months crime
# xgboost does not take in categorical features, need to make the community column categorical

df["community"] = df["community"].astype("category")

x_train, x_test, y_train, y_test = train_test_split(df.loc[:, ["community", "previous_month_graffiti", "previous_month_potholes", "year_previous", "month_previous"]], 
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
    plt.xlabel("previous month potholes")
    plt.ylabel("predicted, true values")
    plt.legend()
    plt.show()
    
results = pd.DataFrame({"n estimators": n_est_list,
                        "rmse": rmselist,
                        "r squared": rsquare_list})

#instantiating model and using early stopping rounds to have algorithm test comparison of validation vs training set rmse
xgb = xgboost.XGBRegressor(max_depth =  5,
                           n_estimators = 4096,
                           objective = "reg:squarederror",
                           booster = "gbtree",
                           random_state = seed,
                           enable_categorical = True,
                           early_stopping_rounds = 100).fit(x_train, y_train, eval_set = [(x_train, y_train), (x_test, y_test)])
xgb_pred = xgb.predict(x_test)

rmse = np.sqrt(mean_squared_error(xgb_pred, y_test))
rsquare = xgb.score(x_test, y_test)

print(f"root mean squared error: {round(rmse, 5)}")
print(f"rsquared: {round(rsquare, 5)}")

print(xgb.best_iteration)
results

final_model = xgboost.XGBRegressor(max_depth = 5,
                                   n_estimators = 41,
                                   objective = "reg:squarederror",
                                   booster = "gbtree",
                                   random_state = seed,
                                   enable_categorical = True).fit(x_train, y_train)

final_model.save_model("final_model.xgb")