import numpy as np
import pandas as pd
import os
from databricks import sql
from dotenv import load_dotenv

load_dotenv()

#querying data from databricks
data_list = ["crime_data", "graffiti_removal", "potholes", "community_areas_boundaries"]
queried_data = []

with sql.connect(server_hostname = os.getenv("DATABRICKS_SERVER_HOSTNAME"),
                 http_path       = os.getenv("DATABRICKS_HTTP_PATH"),
                 access_token    = os.getenv("DATABRICKS_TOKEN")) as connection:

  with connection.cursor() as cursor:
    for i in data_list:
        cursor.execute(f"SELECT * FROM workspace.project_data.{i}")
        rows = cursor.fetchall()
        columns = [col[0] for col in cursor.description]
        queried_data.append(pd.DataFrame(rows, columns=columns))

crime = queried_data[0]
graffiti = queried_data[1]
potholes = queried_data[2]
communities = queried_data[3]

#communities df
communities.columns = communities.columns.str.lower()
communities = communities.rename(columns = {"area_numbe": "community area"})

# crime df

crime.columns = crime.columns.str.lower()
crime["date"] = crime["date"].str[:-11] # to remove irrelevant hourly information
crime["date"] = pd.to_datetime(crime["date"])
crime["month"] = crime["date"].dt.month

# merging community areas (numeric) with community areas (names) to perform groupby
crime = pd.merge(crime, communities[["community", "community area"]], how = "left", on = "community area").drop_duplicates()
crime_groupby = pd.DataFrame(crime.groupby(["year", "month", "community"])["id"].count().reset_index())
crime_groupby = crime_groupby.rename(columns = {"id" : "monthly_count"})
crime_groupby = pd.merge(crime_groupby, communities[["community", 'community area']], how = "left", on = "community")

#merging the count of crimes that occurred within each neighborhood for each month each year back into crimes_df
crime = pd.merge(crime, crime_groupby, how = "left", on = ["year", "month", "community", "community area"]).drop_duplicates()

# graffiti df
graffiti.columns = graffiti.columns.str.lower()
graffiti["creation date"] = pd.to_datetime(graffiti["creation date"], format = "%m/%d/%Y")
graffiti["completion date"] = pd.to_datetime(graffiti["completion date"], format = "%m/%d/%Y")
graffiti["year"] = graffiti["completion date"].dt.year
graffiti["month"] = graffiti["completion date"].dt.month
graffiti = pd.merge(graffiti, communities[["community area", "community"]], how = "left", on = "community area").drop_duplicates()
graffiti_groupby = pd.DataFrame(graffiti.groupby(["year", "month", "community"])["zip code"].count().reset_index())
graffiti_groupby = graffiti_groupby.rename(columns = {"zip code": "monthly_count"})
graffiti = pd.merge(graffiti, graffiti_groupby, how = "left", on = ["year", "month", "community"]).drop_duplicates()

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
os.chdir("c:/Users/AVILA/Desktop")
crime.to_csv("crime.csv")
graffiti.to_csv("graffiti.csv")
potholes.to_csv("potholes.csv")

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
