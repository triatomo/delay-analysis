"""
Exploratory Data Analysis
"""
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

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

""" Number of shipments vs. countries vs. delay """
g=sns.countplot(x='Customer Address Country',hue='Delayed', data=df, order=["AT", "FR", "DE", "ES", "IT", "CH", "LU"])

# put values on the bars
for i in g.patches:
    height = i.get_height()
    g.text(i.get_x() + i.get_width()/2, 200+height, str(i.get_height()), color="black", ha="center")

plt.savefig('Delayed shipments vs. countries.png')
plt.show()

""" Delay probability vs. countries """
g=sns.barplot(x='Customer Address Country', y='Delayed', data=df, ci=None, order=["AT", "FR", "DE", "ES", "IT", "CH", "LU"])

# put values on the bars
for i in g.patches:
    height = i.get_height()
    g.text(i.get_x() + i.get_width()/2, 0.001+height, round(i.get_height(),3), color="black", ha="center")

plt.ylabel('Delay rate')
plt.savefig('Delay rate by countries.png')
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

""" Days vs. delay """

days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

## PPU Days
PPU_values=[] 
for day in days:
    daily_delay = df.apply(lambda x: True if int(x["Delayed"]==True and x['PPU day']==day) else False, axis=1).sum()
    PPU_values.append(daily_delay)   
print(PPU_values)

# Alternative for the for loop above
# for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
#     PPU_values.append(df.apply(lambda row: row['Delayed'] and row ['PPU day']).sum()

PPU_labels = days[:5]            # How to make this automatic?
PPU_values = np.trim_zeros(PPU_values)
plt.title('PPU')
plt.pie(PPU_values, labels=PPU_labels, autopct=lambda p: '{:.1f}%'.format(round(p)) if p > 0 else '', startangle=90)
plt.savefig('PPU day vs. delay.png')
plt.show()

# FHS days
FHS_values=[] 
for day in days:
    daily_delay = df.apply(lambda x: True if int(x["Delayed"]==True and x['FHS day']==day) else False, axis=1).sum()
    FHS_values.append(daily_delay)   
print(FHS_values)

FHS_labels = days[:6] 
FHS_values = np.trim_zeros(FHS_values)
plt.title('FHS')
plt.pie(FHS_values, labels=FHS_labels, autopct='%1.1f%%')
plt.savefig('FHS day vs. delay.png')
plt.show()

# FDA days
FDA_values=[] 
for day in days:
    daily_delay = df.apply(lambda x: True if int(x["Delayed"]==True and x['FDA day']==day) else False, axis=1).sum()
    FDA_values.append(daily_delay)   
print(FDA_values)

FDA_labels = days
FDA_values = np.trim_zeros(FDA_values)
plt.title('FDA')
plt.pie(FDA_values, labels=FDA_labels, autopct='%1.1f%%')
plt.savefig('FDA day vs. delay.png')
plt.show()

"""
Split data into training and test dataset
"""

model_vars = ['Customer Address Country', 'Carrier Company', 'PPU day', 'FHS day', 'FDA day', 'Delivery day', 'PPU-FHS', 'FHS-FDA', 'FDA-Delivery', 'FHS-Delivery', 'Delayed']

shipment_data_relevant = data[model_vars]
shipment_relevant_encoded = pd.get_dummies(shipment_data_relevant)
# # pd.get_dummies(shipment_data_relevant)

x = shipment_relevant_encoded.drop(['Delayed'], axis = 1)
y = shipment_relevant_encoded['Delayed']  

# Creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

""" 
Fix imbalanced class problem by oversampling the data using SMOTE
"""

sm = SMOTE(random_state=12, ratio=1.0)
sm_x_train, sm_y_train = sm.fit_sample(x_train, y_train)
# Why making a data frame necessary?
sm_y_train = pd.DataFrame(data = sm_y_train, columns= ['Delayed'])
columns = x_train.columns


print('Length of oversampled data is ', len(sm_x_train))
print('Number of delayed shipments in oversampled data is ', len(sm_y_train[sm_y_train['Delayed']==1]))
print('Number of on time shipments is ', len(sm_y_train[sm_y_train['Delayed']==0]))
print('Proportion of delayed shipments is ', len(sm_y_train[sm_y_train['Delayed']==1])/len(sm_x_train))
print('Proportion of on time shipments is', len(sm_y_train[sm_y_train['Delayed']==0])/len(sm_x_train))
# LogReg
# Randon Forest
# NN
# kNN
