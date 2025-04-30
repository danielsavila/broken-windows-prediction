import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

path = "c:/Users/danie/Desktop"
os.chdir(path)

crime = pd.read_csv("crime.csv")
graffiti = pd.read_csv("graffiti.csv")
potholes = pd.read_csv("potholes.csv")

labels = graffiti[["year", "month", "community", "monthly_count"]]
labels = labels.rename(columns = {"monthly_count": "monthly_count_graffiti"}).drop_duplicates()
labels = pd.merge(labels, potholes[["year", "month", "community", "monthly_count"]], how = "inner", on = ["year", "month", "community"])
labels = labels.rename(columns = {"monthly_count":"monthly_count_potholes"}).drop_duplicates().sort_values(by = "year")
labels = pd.merge(labels, crime[["year", "month", "community", "monthly_count"]], how = "inner", on = ["year", "month", "community"]).drop_duplicates()
labels = labels.rename(columns = {"monthly_count":"monthly_count_crime"}).sort_values(by = ["year", "month"], ascending = True)
labels.drop("date", inplace = True, axis = 1)

labels.to_csv("monthly_count_df.csv")

#having trouble shifting the columns so that they match the columns right
labels["date"] = pd.to_datetime(labels["year"].astype(str) + '-' + labels["month"].astype(str)) + pd.DateOffset(months = 1)


# we are predicting the future months' crime, so we push each month in the crime df one month ahead
# and month 13 becomes 1

plt.scatter(x = labels["monthly_count_graffiti"], y = labels["monthly_count_crime"])
plt.xlabel("monthly graffiti")
plt.ylabel("monthly crime")
plt.show()

plt.scatter(labels["monthly_count_potholes"], labels["monthly_count_crime"])
plt.show()