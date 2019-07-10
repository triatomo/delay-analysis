"""
Exploratory Data Analysis
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_data.csv', dtype={'Delayed': np.bool})
df['all'] = "" 

""" PPU-FHS vs. Delay """
q99 = df["PPU-FHS"].quantile(0.99)

plt.subplot(2,1,1)
plt.title("PPU-FHS")
g = sns.violinplot(y='PPU-FHS', x='all', hue='Delayed', split=True, data=df[df['PPU-FHS']>=q99])
plt.yscale('log', basey=2)
plt.ylabel('Days')
plt.xlabel(' ')

plt.subplot(2,1,2)
g = sns.violinplot(y='PPU-FHS', x='all', hue='Delayed', split=True, data=df[df['PPU-FHS']<q99])
plt.ylabel('Days')
plt.xlabel(' ')

# plt.savefig('PPU-FHS vs. delay.png')
plt.show()

""" FHS-FDA vs. Delay """
q99 = df["FHS-FDA"].quantile(0.99)

plt.subplot(2,1,1)
plt.title("FHS-FDA")
g = sns.violinplot(y='FHS-FDA', x='all', hue='Delayed', split=True, data=df[df['FHS-FDA']>=q99])
plt.yscale('log', basey=2)
plt.ylabel('Days')
plt.xlabel(' ')

plt.subplot(2,1,2)
g = sns.violinplot(y='FHS-FDA', x='all', hue='Delayed', split=True, data=df[df['FHS-FDA']<q99])
plt.ylabel('Days')
plt.xlabel(' ')

plt.savefig('FHS-FDA vs. delay.png')
plt.show()

""" FHS-Delivery vs. Delay """
q99 = df["FHS-Delivery"].quantile(0.99)

plt.subplot(2,1,1)
plt.title("FHS-Delivery")
g = sns.violinplot(y='FHS-Delivery', x='all', hue='Delayed', split=True, data=df[df['FHS-Delivery']>=q99])
plt.yscale('log', basey=2)
plt.ylabel('Days')
plt.xlabel(' ')

plt.subplot(2,1,2)
g = sns.violinplot(y='FHS-Delivery', x='all', hue='Delayed', split=True, data=df[df['FHS-Delivery']<q99])
plt.ylabel('Days')
plt.xlabel(' ')

plt.savefig('FHS-Delivery vs. delay.png')
plt.show()

""" FDA-Delivery vs. Delay """
q99 = df["FDA-Delivery"].quantile(0.99)

plt.subplot(2,1,1)
plt.title("FDA-Delivery")
g = sns.violinplot(y='FDA-Delivery', x='all', hue='Delayed', split=True, data=df[df['FDA-Delivery']>=q99])
plt.yscale('log', basey=2)
plt.ylabel('Days')
plt.xlabel(' ')

plt.subplot(2,1,2)
g = sns.violinplot(y='FDA-Delivery', x='all', hue='Delayed', split=True, data=df[df['FDA-Delivery']<q99])
plt.ylabel('Days')
plt.xlabel(' ')

plt.savefig('FDA-Delivery vs. delay.png')
plt.show()

""" Carrier Company vs. Delay """
g=sns.barplot(x='Carrier', y='Delayed', data=df, ci=None)

# change labels on x axis
N = 10
ind = np.arange(N)
plt.xticks(ind, ('C - AT', 'B - FR', 'D - FR', 'A - DE', 'E - ES', 'G - FR', 'H - CH', 'F - IT', 'A - LU', 'A - NL'))
g.set_xticklabels(g.get_xticklabels(), rotation=70, ha="right", fontsize=7)
# put values on the bars
for i in g.patches:
    height = i.get_height()
    g.text(i.get_x() + i.get_width()/2, 0.001+height, round(i.get_height(),3), color="black", ha="center")

plt.ylabel('Delay rate')
plt.tight_layout()  
plt.savefig('carrier vs. delay.png')
plt.show()