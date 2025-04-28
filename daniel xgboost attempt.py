import numpy as np
import pandas as pd
import os

path = "c:/Users/danie/Desktop"
os.chdir(path)

crime = pd.read_csv("crime.csv")
graffiti = pd.read_csv("graffiti.csv")
potholes = pd.read_csv("potholes.csv")

df = graffiti[["year", "month", "community", "monthly_count"]]
df = df.rename(columns = {"monthly_count": "monthly_count_graffiti"}).drop_duplicates()
df = pd.merge(df, potholes[["year", "month", "community", "monthly_count"]], how = "inner", on = ["year", "month", "community"])
df = df.rename(columns = {"monthly_count":"monthly_count_potholes"}).drop_duplicates()
df.head()