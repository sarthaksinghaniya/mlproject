import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
import seaborn.objects as so

df = sns.load_dataset("penguins")
sns.pairplot(df, hue="species")
'''
print(sns.get_dataset_names())
print("Available datasets:", sns.get_dataset_names())       
'''
df = sns.load_dataset("brain_networks")

