import pickle
import pandas as pd
import os

domain = "sc"
path = "results/" + domain

lst = []
for folder in os.listdir(path):
    approach = path + "/" + folder
    tmp = []
    for file in os.listdir(approach):
        instance = approach + "/" + file
        with open(instance, "rb") as fp:
            b = pickle.load(fp)
        tmp.append(b[0][-1])
    lst.append(tmp)

df = pd.DataFrame(lst).T
df.columns = os.listdir(path)
df.index = os.listdir(approach)
df.to_csv(domain + ".csv")
