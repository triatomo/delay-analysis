import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cleaned_data.csv', dtype={'Delayed': np.bool})

""" Number of shipments vs. countries vs. delay """
g=sns.countplot(x='Customer Address Country',hue='Delayed', data=df, order=["AT", "FR", "DE", "ES", "IT", "CH"])

# put values on the bars
for i in g.patches:
    height = i.get_height()
    g.text(i.get_x() + i.get_width()/2, 200+height, str(i.get_height()), color="black", ha="center")

plt.savefig('Delayed shipments vs. countries.png')
plt.show()

""" Delay probability vs. countries """
g=sns.barplot(x='Customer Address Country', y='Delayed', data=df, ci=None, order=["AT", "FR", "DE", "ES", "IT", "CH"])

# put values on the bars
for i in g.patches:
    height = i.get_height()
    g.text(i.get_x() + i.get_width()/2, 0.001+height, round(i.get_height(),3), color="black", ha="center")

plt.ylabel('Delay probabibility')
plt.savefig('Delay probability by 6 major countries.png')
plt.show()