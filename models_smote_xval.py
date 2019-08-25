import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier      # NeighborhoodComponentsAnalysis cannot import
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from imblearn.pipeline import make_pipeline

df = pd.read_csv('cleaned_data.csv', dtype={'Delayed': np.bool})
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 4)

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

# Scale data to feed prediction models
print('Scaling data...')
scaler = StandardScaler().fit(x)
x_scale = scaler.transform(x)

""" 
Fix imbalanced class problem by oversampling the data using SMOTE
"""
# Oversampling under represented classes with SMOTE
print('Oversampling under-represented data...')
sm = SMOTE('not majority', random_state=42)
sm_x, sm_y = sm.fit_sample(x_scale, y)

df_y_train = pd.DataFrame(data = sm_y, columns= ['Delayed'])
print('Length of oversampled data is', len(sm_x))     #62364 
print('Number of delayed shipments in oversampled data is', len(sm_y[df_y_train['Delayed']==True]))  #31182
print('Number of on time shipments is', len(sm_y[df_y_train['Delayed']==False]))  #31182

"""
Building prediction models
"""
random_state = 1        # Fixate a random state so that the results are reproducible

logreg = LogisticRegression(random_state=random_state)
rf = RandomForestClassifier(random_state=random_state)
mlp = MLPClassifier(hidden_layer_sizes= (50,50,50), max_iter=100)
cart = DecisionTreeClassifier(random_state=random_state)
knn = KNeighborsClassifier()
svm = SVC(random_state=random_state)

clf = []        # List of algorithms
clf.append(logreg)
clf.append(rf)
clf.append(mlp)
clf.append(cart)
clf.append(knn)
clf.append(svm)

"""Show the predictive power of the models by cross validating using stratified k fold
"""
kfold = StratifiedKFold(n_splits=5)
cross_val_results_sm = []      # Returns n-fold results of cross validation of each predictor
for classifier in clf:
    smoted_classifier = make_pipeline(sm, classifier)
    cv_score = cross_val_score(smoted_classifier, x_scale, y, scoring='accuracy', cv=kfold)
    cross_val_results_sm.append(cv_score)

cv_means_sm = []    # Returns the means of the n-fold cross val results
cv_std_sm = []      # Returns the standard deviation of the n-fold cross val results
for cv in cross_val_results:
    cv_means_sm.append(cv.mean())
    cv_std_sm.append(cv.std())   

cv_res_sm = pd.DataFrame({"Cross Val Means":cv_means_sm, "Cross Val Errors":cv_std_sm, "Algorithm":["Logistic Regression", "Random Forest", "MLP", "CART", "KNN", "SVM"]})

order = cv_res.sort_values('Cross Val Means')      # Order bars in ascending order
g = sns.barplot("Cross Val Means","Algorithm",data = cv_res_sm, order=order['Algorithm'], palette="Set3",orient = "h", **{'xerr':cv_std_sm})

for i in g.patches:         # Put labels on bars
    width = i.get_width()-(i.get_width()*0.12)        # Put labels left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross Validation Score by SMOTE")
plt.tight_layout()  
plt.savefig('Accuracy scores_kfold_sm.png')
plt.show()