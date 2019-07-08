"""
Exploratory Data Analysis
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_data.csv', dtype={'Delayed': np.bool})
pd.isnull(df['Delayed']).values.any()
df['all'] = "" 

q99 = df["PPU-FHS"].quantile(0.99)

plt.subplot(2,1,1)
plt.title("PPU-FHS")
g = sns.violinplot(y='PPU-FHS', x='all', hue='Delayed', split=True, data=df[df['PPU-FHS']>=q99])
plt.yscale('log', basey=2)
plt.ylabel('Days')

plt.subplot(2,1,2)
g = sns.violinplot(y='PPU-FHS', x='all', hue='Delayed', split=True, data=df[df['PPU-FHS']<q99])

plt.ylabel('Days')
plt.show()
