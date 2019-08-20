import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss
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
Split data into training and test dataset
"""

# Define columns relevant for the prediction
model_vars = ['Customer Address Country', 'Carrier', 'PPU day', 'FHS day', 'FDA day', 'Delivery day', 'PPU-FHS', 'FHS-FDA', 'FDA-Delivery', 'FHS-Delivery', 'Delayed']

rel_data = df[model_vars]
rel_data_encoded = pd.get_dummies(rel_data)     # convert categorical vars into numerical.Yields 56 cols

# Separate predictor from target variable
x = rel_data_encoded.drop(['Delayed'], axis = 1)       # predictor vars
y = rel_data_encoded['Delayed']                        # target variable

# Creating training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

# Undersampling with NearMiss
print('Under-sampiling over-represented data...')
nm = NearMiss('not minority', random_state=42)
nm_x_train, nm_y_train = nm.fit_sample(x_train, y_train)
nm_x_test, nm_y_test = nm.fit_sample(x_test, y_test)

df_y_train = pd.DataFrame(data = nm_y_train, columns= ['Delayed'])
print('Length of under-sampled data is', len(nm_x_train))   # 4668
print('Number of delayed shipments in oversampled data is', len(nm_y_train[df_y_train['Delayed']==True]))   # 2334
print('Number of on time shipments is', len(nm_y_train[df_y_train['Delayed']==False]))  # 2334

# Scale data to feed prediction models
print('Scaling data...')
scaler = StandardScaler().fit(nm_x_train)

nm_x_train = scaler.transform(nm_x_train)
nm_x_test = scaler.transform(nm_x_test)

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
    classifier.fit(nm_x_train, nm_y_train)
    y_pred.append(classifier.predict(nm_x_test))
print(y_pred)

accuracy = []       # List of prediction accuracies by algorithm
for pred in y_pred:
    accuracy.append(accuracy_score(nm_y_test, pred))
print(accuracy)

# df of prediction results above
pred_res = pd.DataFrame({"Accuracy Score":accuracy, "Algorithm":["Logistics Regression", "Random Forest", "MLP"]})

order = pred_res.sort_values('Accuracy Score')      # Order bars in ascending order
g = sns.barplot("Accuracy Score","Algorithm",data = pred_res, order=order['Algorithm'], palette="Set3",orient = "h")

for i in g.patches:         # Put labels on bars
    width = i.get_width()-0.08        # Put labels -0.14 left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Accuracy Score")
g = g.set_title("Accuracy Score by NearMiss")
plt.tight_layout()  
plt.savefig('Accuracy scores before cross val_nm.png')
plt.show()
