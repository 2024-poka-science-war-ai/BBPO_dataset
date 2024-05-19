"""
File to show result of learning from the csv file.
"""

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style("whitegrid")

df = pd.read_csv("log_melee.csv")
sns.lineplot(data=df, x='episode_id', y='score')
plt.show()
