import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt 

# I first grab some data from seaborn and make an extra column so that there
# are exactly 8 columns for our 8 axes
data = sns.load_dataset('car_crashes')
data = data.drop('abbrev', axis=1)
data['total2'] = data['total'] * 2

# Set figsize here
fig, axes = plt.subplots(nrows=2, ncols=4, figsize=(12,5))

# if you didn't set the figsize above you can do the following
# fig.set_size_inches(12, 5)

# flatten axes for easy iterating
for i, ax in enumerate(axes.flatten()):
    sns.boxplot(x= data.iloc[:, i],  orient='v' , ax=ax)

fig.tight_layout()

plt.show()