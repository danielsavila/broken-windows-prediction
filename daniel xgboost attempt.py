import numpy as np
import pandas as pd
import os

path = "c:/Users/danie/Desktop"
os.chdir(path)

crime = pd.read_csv("crime.csv")
graffiti = pd.read_csv("graffiti.csv")
potholes = pd.read_csv("potholes.csv")

