
import numpy as np
import pandas as pd
import os

os.chdir("C:/Users/danie/onedrive/documents/github/broken-windows-prediction")
crime2014 = pd.read_csv("Crime 2014.csv")
crime2015 = pd.read_csv("Crime 2015.csv")
crime2016 = pd.read_csv("Crimes 2016.csv")
crime2017 = pd.read_csv("Crimes 2017.csv")
crime2018 = pd.read_csv('Crimes 2018.csv')
graffiti1 = pd.read_csv("graffiti removal 2014 - 2015.csv")
graffiti2 = pd.read_csv("graffiti removal 2016 - 2018.csv")
potholes = pd.read_csv("potholes.csv")