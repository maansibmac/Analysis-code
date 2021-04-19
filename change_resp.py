# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import math
Z = norm.ppf
from classes_text import *
import os

# table_data = pd.read_csv(f"../../data/test_29032021_1/data/data_all.csv")
# print(table_data)
# # %%
# new_resp = [9, 27, 28, 41, 65, 74]

# for i in new_resp:
#     x = table_data[table_data['Trial'] == i].index.values
#     table_data.loc[x, 'Responses'] = 0
#     print(table_data.loc[x, 'Responses'])

# table_data.to_csv(f"/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/test_29032021_1/data/data_all.csv", index = False)

# %%
new_resp_2 = [98]

another_data = pd.read_csv(f"../../data/test_01042021_2/data/data_all.csv")

for y in new_resp_2:
    z = another_data[another_data['Trial'] == y].index.values
    another_data.loc[z, 'Responses'] = 0
    print(another_data.loc[z, 'Responses'])

table_data.to_csv(f"/Users/manny/Documents/ProjectCold/expt4_py_nontactileColdnew/data/test_01042021_2/data/data_all.csv", index = False)




