import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

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

labels.to_csv("monthly_count_df.csv")

# note that there is some missingness in the data, as identified below.
