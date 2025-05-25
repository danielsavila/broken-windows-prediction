
import numpy as np
import pandas as pd
import os


# replace with your local github repo path
github_path = "C:/Users/danie/onedrive/documents/github/broken-windows-prediction/raw-data"
desktop_path = "C:/Users/danie/Desktop"

os.chdir(github_path)

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

crime_df["date"] = crime_df["date"].str[:-11] # to remove irrelevant time information
crime_df["date"] = pd.to_datetime(crime_df["date"])
crime_df["month"] = crime_df["date"].dt.month

# merging community areas (numeric) with community areas (names) to perform groupby
crime_df = pd.merge(crime_df, communities[["community", "community area"]], how = "left", on = "community area").drop_duplicates()
crime_groupby = pd.DataFrame(crime_df.groupby(["year", "month", "community"])["id"].count().reset_index())
crime_groupby = crime_groupby.rename(columns = {"id" : "monthly_count"})
crime_groupby = pd.merge(crime_groupby, communities[["community", 'community area']], how = "left", on = "community")

#merging the count of crimes that occurred within each neighborhood for each month each year back into crimes_df
crime_df = pd.merge(crime_df, crime_groupby, how = "left", on = ["year", "month", "community", "community area"]).drop_duplicates()
crime_df.head()


# graffiti df
graffiti_df = pd.concat([graffiti1, graffiti2], axis = 0)
graffiti_df.columns = graffiti_df.columns.str.lower()
graffiti_df["creation date"] = pd.to_datetime(graffiti_df["creation date"], format = "%m/%d/%Y")
graffiti_df["completion date"] = pd.to_datetime(graffiti_df["completion date"], format = "%m/%d/%Y")
graffiti_df["year"] = graffiti_df["completion date"].dt.year
graffiti_df["month"] = graffiti_df["completion date"].dt.month
graffiti_df = pd.merge(graffiti_df, communities[["community area", "community"]], how = "left", on = "community area").drop_duplicates()
graffiti_groupby = pd.DataFrame(graffiti_df.groupby(["year", "month", "community"])["zip code"].count().reset_index())
graffiti_groupby = graffiti_groupby.rename(columns = {"zip code": "monthly_count"})
graffiti_df = pd.merge(graffiti_df, graffiti_groupby, how = "left", on = ["year", "month", "community"]).drop_duplicates()
graffiti_df.head()

#potholes df
potholes.columns = potholes.columns.str.lower()
potholes = pd.merge(potholes, communities[["community area", "community"]], how = "left", on = "community area").drop_duplicates()
potholes["completion date"] = pd.to_datetime(potholes["completion date"])
potholes["completion date"] = pd.to_datetime(potholes["creation date"])
potholes["month"] = potholes["completion date"].dt.month
potholes["year"] = potholes["completion date"].dt.year
potholes_groupby = pd.DataFrame(potholes.groupby(["year", "month", "community"])["creation date"].count().reset_index())
potholes_groupby = potholes_groupby.rename(columns = {"creation date": "monthly_count"})
potholes = pd.merge(potholes, potholes_groupby, how = "left", on = ["year", "month", "community"]).drop_duplicates()

#exporting cleaned datasets to desktop
#fyi that these end up being massive files
os.chdir(desktop_path)
crime_df.to_csv("crime.csv")
graffiti_df.to_csv("graffiti.csv")
potholes.to_csv("potholes.csv")
