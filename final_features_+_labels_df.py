import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime
import dateutil.relativedelta


path = "c:/Users/danie/Desktop"
os.chdir(path)

crime = pd.read_csv("crime.csv")
graffiti = pd.read_csv("graffiti.csv")
potholes = pd.read_csv("potholes.csv")

#combining the three dataframes to create the monthly count df
labels = graffiti[["year", "month", "community", "monthly_count"]]
labels = labels.rename(columns = {"monthly_count": "monthly_count_graffiti"}).drop_duplicates()
labels = pd.merge(labels, potholes[["year", "month", "community", "monthly_count"]], how = "inner", on = ["year", "month", "community"])
labels = labels.rename(columns = {"monthly_count":"monthly_count_potholes"}).drop_duplicates().sort_values(by = "year")
labels = pd.merge(labels, crime[["year", "month", "community", "monthly_count"]], how = "inner", on = ["year", "month", "community"]).drop_duplicates()
labels = labels.rename(columns = {"monthly_count":"monthly_count_crime"}).sort_values(by = ["year", "month"], ascending = True)


#had to do some fixing of the data to get the previous month, potholes, and graffiti cleanup
copy = pd.DataFrame(labels).reset_index()

list_date = []
for i in range(len(copy)):
    list_date.append(f"{copy["year"][i]}" + "-" f"{copy["month"][i]}" + "-" +  "01")
    
copy["month_copy"] = pd.to_datetime(list_date)
copy["true_month"] = copy["month_copy"]
copy = copy.rename(columns = {"monthly_count_crime" : "realized_crime", "month_copy": "previous_month"}) 

month_df = copy[["realized_crime", "previous_month", "true_month", "community"]]
copy = copy.drop(["realized_crime", "true_month", "year", "month", "index"], axis = 1)   
month_df.loc[:, "previous_month"] = month_df.loc[:, "previous_month"] - pd.DateOffset(months = 1)

copy = pd.merge(copy, month_df, how = "inner", on = ["previous_month", "community"])
copy = copy.rename(columns = {"monthly_count_graffiti": "previous_month_graffiti", "monthly_count_potholes":"previous_month_potholes"})
copy["year_previous"] = copy["previous_month"].dt.year
copy["month_previous"] = copy["previous_month"].dt.month
copy = copy.iloc[:, [0, 1, 2, 3, 6, 7, 4, 5]]
copy.head()
labels = copy

labels.to_csv("monthly_count_df.csv")
