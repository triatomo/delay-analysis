import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV

df = pd.read_csv('cleaned_data.csv', dtype={'Delayed': np.bool})
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 4)

"""
Exploratory Data Analysis
"""
print('Number of shipments total:', df['Delayed'].count())      # 50024
print("Number of delayed shipments:", df['Delayed'].value_counts()[True])       # 3427
print("Number of on time shipments:", df['Delayed'].value_counts()[False])      # 46597

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

# change labels on x axis due to data protection
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

# Define columns relevant for the prediction
model_vars = ['Customer Address Country', 'Carrier', 'PPU day', 'FHS day', 'FDA day', 'Delivery day', 'PPU-FHS', 'FHS-FDA', 'FDA-Delivery', 'FHS-Delivery', 'Delayed']

rel_data = df[model_vars]
rel_data_encoded = pd.get_dummies(rel_data)     # convert categorical vars into numerical.Yields 56 cols

x = rel_data_encoded.drop(['Delayed'], axis = 1)       # predictor vars
y = rel_data_encoded['Delayed']                        # target variable

# Creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

""" 
Fix imbalanced class problem by oversampling the data using SMOTE
"""
# Oversampling under represented classes with SMOTE
print('Oversampling under-represented data...')
sm = SMOTE('not majority')
sm_x_train, sm_y_train = sm.fit_sample(x_train, y_train)
sm_x_test, sm_y_test = sm.fit_sample(x_test, y_test)

df_y_train = pd.DataFrame(data = sm_y_train, columns= ['Delayed'])
print('Length of oversampled data is', len(sm_x_train))
print('Number of delayed shipments in oversampled data is', len(sm_y_train[df_y_train['Delayed']==True]))
print('Number of on time shipments is', len(sm_y_train[df_y_train['Delayed']==False]))
print('Proportion of delayed shipments is', len(sm_y_train[df_y_train['Delayed']==True])/len(sm_x_train))
print('Proportion of on time shipments is', len(sm_y_train[df_y_train['Delayed']==False])/len(sm_x_train))

# Scale data to feed prediction models
print('Scaling data...')
scaler = StandardScaler().fit(sm_x_train)

sm_x_train = scaler.transform(sm_x_train)
sm_x_test = scaler.transform(sm_x_test)

"""
Building prediction models
"""
random_state = 1        # Fixate a random state so that the results are reproducible

logreg = LogisticRegression(random_state=random_state)
rf = RandomForestClassifier(random_state=random_state)
mlp = MLPClassifier(hidden_layer_sizes= (50,50,50), max_iter=100)

clf = []        # List of algorithms
clf.append(logreg)
clf.append(rf)
clf.append(mlp)

print('Training prediction models (this could take some time)...')
y_pred = []         # List of prediction results
for classifier in clf:
    classifier.fit(sm_x_train, sm_y_train)
    y_pred.append(classifier.predict(sm_x_test))
    accuracy_score.append(accuracy_score.(sm_y_train, y_pred))

accuracy = []       # List of prediction accuracies by algorithm
for pred in y_pred:
    accuracy.append(accuracy_score(sm_y_test, pred))
print(accuracy)

# df of prediction results above
pred_res = pd.DataFrame({"Accuracy Score":accuracy, "Algorithm":["Logistics Regression", "Random Forest", "MLP"]})

order = pred_res.sort_values('Accuracy Score')      # Order bars in ascending order
g = sns.barplot("Accuracy Score","Algorithm",data = pred_res, order=order['Algorithm'], palette="Set3",orient = "h")

for i in g.patches:         # Put labels on bars
    width = i.get_width()/i.get_width()-0.14        # Put labels -0.14 left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Accuracy Score")
g = g.set_title("Accuracy Score")
plt.tight_layout()  
plt.savefig('Accuracy scores before cross val.png')
plt.show()

"""
Show the power of the model with cross validation
"""
kfold = StratifiedKFold(n_splits=5)
cross_val_results = []      # Returns n-fold results of cross validation of each predictor
for classifier in clf:
    cross_val_results.append(cross_val_score(classifier, sm_x_train, sm_y_train, scoring='accuracy', cv=kfold))

cv_means = []    # Returns the means of the n-fold cross val results
cv_std = []      # Returns the standard deviation of the n-fold cross val results
for cv in cross_val_results:
    cv_means.append(cv.mean())
    cv_std.append(cv.std())   

cv_res = pd.DataFrame({"Cross Val Means":cv_means, "Cross Val Errors":cv_std, "Algorithm":["Logistics Regression", "Random Forest", "MLP"]})

order = cv_res.sort_values('Cross Val Means')      # Order bars in ascending order
g = sns.barplot("Cross Val Means","Algorithm",data = cv_res, order=order['Algorithm'], palette="Set3",orient = "h", **{'xerr':cv_std})

for i in g.patches:         # Put labels on bars
    width = i.get_width()-(i.get_width()*0.12)        # Put labels -0.14 left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross Validation Scores")
plt.tight_layout()  
plt.savefig('Accuracy scores after cross val.png')
plt.show()



# Evaluating NN model and visualization
y_pred = y_pred.astype(np.int64)
y_pred = le.inverse_transform(y_pred)

sm_y_test = sm_y_test.astype(np.int64)
sm_y_test = le.inverse_transform(sm_y_test)

cm = confusion_matrix(sm_y_test, y_pred)
cm_norm = cm/cm.astype(np.float).sum(axis=1)
print('Confusion matrix:')
print(cm)
sns.heatmap(cm_norm, center=0.5,
            annot=True, fmt='.2f',
            vmin=0, vmax=1, cmap='Reds',
            xticklabels=['A','B','C','D','E'], 
            yticklabels=['A','B','C','D','E'])
plt.savefig('cm_norm_heatmap.png')
print('Saving normalized confusion matrix heatmap to "cm_norm_heatmap.png"')

print('Classification matrix:')
print(classification_report(sm_y_test,y_pred))
# kNN
