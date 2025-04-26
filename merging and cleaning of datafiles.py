
import numpy as np
import pandas as pd
import os

# replace with your local github repo path
path = "C:/Users/danie/onedrive/documents/github/broken-windows-prediction"

os.chdir(path)

crime2014 = pd.read_csv("Crime 2014.csv")
crime2015 = pd.read_csv("Crimes 2015.csv")
crime2016 = pd.read_csv("Crimes 2016.csv")
crime2017 = pd.read_csv("Crimes 2017.csv")
crime2018 = pd.read_csv('Crimes 2018.csv')
graffiti1 = pd.read_csv("graffiti removal 2014 - 2015.csv")
graffiti2 = pd.read_csv("graffiti removal 2016 - 2018.csv")
potholes = pd.read_csv("potholes.csv")
communities = pd.read_csv("community areas boundaries.csv")

#communities df
communities.columns = communities.columns.str.lower()
communities = communities.rename(columns = {"area_numbe": "community area"})

# crime df
crime_list = [crime2014, crime2015, crime2016, crime2017, crime2018]
crime_df = pd.DataFrame()
for i in range(len(crime_list)):
    crime_df = pd.concat([crime_df, crime_list[i]], axis = 0)

crime_df.columns = crime_df.columns.str.lower()

crime_df["date"] = crime_df["date"].str[:-11]
crime_df["date"] = pd.to_datetime(crime_df["date"])
crime_df["month"] = crime_df["date"].dt.month

# merging community areas (numeric) with community areas (names) to perform groupby
crime_df = pd.merge(crime_df, communities[["community", "community area"]], how = "left", on = "community area")
crime_groupby = pd.DataFrame(crime_df.groupby(["year", "month", "community"])["id"].count().reset_index())
crime_groupby = crime_groupby.rename(columns = {"id" : "count"})
crime_groupby.head()



# graffiti df
graffiti_df = pd.concat([graffiti1, graffiti2], axis = 0)
graffiti_df.columns = graffiti_df.columns.str.lower()
graffiti_df.info()
graffiti_df["creation date"] = pd.to_datetime(graffiti_df["creation date"], format = "%m/%d/%Y")
graffiti_df["completion date"] = pd.to_datetime(graffiti_df["completion date"], format = "%m/%d/%Y")

