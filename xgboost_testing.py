import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import xgboost
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from databricks import sql

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
# xgboost does not take in categorical features, used enable_categorical experimental feature

df["community"] = df["community"].astype("category")

x_train, x_test, y_train, y_test = train_test_split(df.loc[:, ["community", "previous_month_graffiti", "previous_month_potholes", "year_previous", "month_previous"]], 
                                                    df["realized_crime"], 
                                                    train_size = .7, 
                                                    random_state = seed, 
                                                    shuffle = True)

#now running cross validation to determine best number of trees for this model

param = {"max_depth": 4, "grow_policy": 
         "lossguide", "learning_rate": .1, 
         "objective": "reg:squarederror", 
         "random_state": seed, 
         "booster": "gbtree"}

num_rounds = 150

train_data = xgboost.DMatrix(x_train, label = y_train, enable_categorical = True)
xgb_cv = xgboost.cv(params = param,
       dtrain = train_data,
       num_boost_round = num_rounds,
       nfold = 5,
       metrics = {"rmse"})

xgb_cv

print(xgb_cv["test-rmse-mean"].idxmin()) #number of optimal boosting rounds = 130

#using optimal number of boosting rounds in final model
final_model = xgboost.XGBRegressor(max_depth = 4,
                                   n_estimators = 130,
                                   objective = "reg:squarederror",
                                   booster = "gbtree",
                                   random_state = seed,
                                   enable_categorical = True).fit(x_train, y_train)

pred_values = final_model.predict(x_test)
test_rmse = np.sqrt(mean_squared_error(pred_values, y_test))

test_rmse # +- 32.98 crimes per month, not bad!

#final_model.save_model("final_model.xgb")