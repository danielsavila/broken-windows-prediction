import os
import torch
from torch import functional as f
from torchinfo import summary
from torch import nn, optim
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost_testing import df

#setting up environmental variables for rnn
#torch.use_deterministic_algorithms(True) #for reproducability
#torch.backends.cudnn.benchmark = False
#torch.backends.cudnn.deterministic = True
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE' #for known np package error

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
seed = 123456

scaler = StandardScaler()
def df_to_tensor(df):
    df = pd.DataFrame(df)
    if "community" in df.columns:
        df = df.drop("community", axis = 1)
    
    df = scaler.fit_transform(df)
    df = torch.Tensor(df).float()
    return df

def create_sequences(data, community_ids, seq_length = 3):
    ds, comm_ids = [], []
    for i in range(len(data) - seq_length):
        ds.append(data[i: i + seq_length])
        comm_ids.append(community_ids[i + seq_length])
    return torch.stack(ds), torch.stack(comm_ids)


#organizing df by community, then by year and month as prep for rnn
df = df.groupby("community", group_keys=True, observed = True
                ).apply(lambda df: df.sort_values(["year_previous", "month_previous"]), include_groups = False
                ).reset_index().drop("level_1", axis = 1)

#inputting missing values by creating empty df and merging with data
#this is so rnn has consistent batch sizes later
mean_crime = df.groupby(["community", "year_previous"])[["realized_crime", "previous_month_graffiti", "previous_month_potholes"]].mean().reset_index()
communities = mean_crime["community"].unique()
years = mean_crime["year_previous"].unique()
months = np.arange(1,13)

base_df = pd.MultiIndex.from_product([communities, years, months], names = ['community', "year_previous", "month_previous"]).to_frame(index = False)
base_df = base_df.merge(mean_crime, how = "left", on = ["community", "year_previous"])
base_df[["realized_crime", "previous_month_graffiti", "previous_month_potholes"]] = base_df[["realized_crime", "previous_month_graffiti", "previous_month_potholes"]].fillna(0)
df = base_df.merge(df, how = "left", on = ["community", "year_previous", "month_previous"])
df["realized_crime_y"] = df["realized_crime_y"].combine_first(df["realized_crime_x"])
df["previous_month_graffiti_y"] = df["previous_month_graffiti_y"].combine_first(df["previous_month_graffiti_x"])
df["previous_month_potholes_y"] = df["previous_month_potholes_y"].combine_first(df["previous_month_graffiti_x"])
df = df.drop(["realized_crime_x", 
              "previous_month_graffiti_x", 
              "previous_month_potholes_x"], axis = 1
              ).rename(
                        {"realized_crime_y":"realized_crime",
                        "previous_month_graffiti_y": "previous_month_graffiti",
                        "previous_month_potholes_y": "previous_month_potholes"}, axis = 1)

#predicting 2018 using 2014 - 2017
train = df[df["year_previous"].isin([2014, 2015, 2016, 2017])]
test = df[df["year_previous"] == 2018]
x_train = train.loc[:, ["community", "previous_month_graffiti", "previous_month_potholes", "year_previous", "month_previous"]]
x_test = test.loc[:, ["community", "previous_month_graffiti", "previous_month_potholes", "year_previous", "month_previous"]]
y_train = train.loc[:,"realized_crime"]
y_test = test.loc[:, "realized_crime"]


#some quick data preprocessing to remove community categorical, reintroduce later
communities = x_train["community"]
communities_unique = sorted(communities.unique())
community_to_id = {name: idx for idx, name in enumerate(communities_unique)}
community_ids = communities.map(community_to_id).values


x_train = df_to_tensor(x_train)
y_train = df_to_tensor(y_train)
x_test = df_to_tensor(x_test)
y_test = df_to_tensor(y_test)
community_tensor = torch.tensor(community_ids, dtype=torch.long)