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
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier      # NeighborhoodComponentsAnalysis cannot import
from sklearn.svm import SVC
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

df_y_train = pd.DataFrame(data = nm_y_train, columns= ['Delayed'])
print('Length of under-sampled data is', len(nm_x_train))   # 4668
print('Number of delayed shipments in oversampled data is', len(nm_y_train[df_y_train['Delayed']==True]))   # 2334
print('Number of on time shipments is', len(nm_y_train[df_y_train['Delayed']==False]))  # 2334

# Scale data to feed prediction models
print('Scaling data...')
scaler = StandardScaler().fit(nm_x_train)

nm_x_train = scaler.transform(nm_x_train)
x_test = scaler.transform(x_test)

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

print('Training prediction models (this could take some time)...')
y_pred = []         # List of prediction results
for classifier in clf:
    classifier.fit(nm_x_train, nm_y_train)
    y_pred.append(classifier.predict(x_test))
print(y_pred)

accuracy = []       # List of prediction accuracies by algorithm
for pred in y_pred:
    accuracy.append(accuracy_score(y_test, pred))
print(accuracy)

# df of prediction results above
pred_res = pd.DataFrame({"Accuracy Score":accuracy, "Algorithm":["Logistic Regression", "Random Forest", "MLP", "CART", "KNN", "SVM"]})

order = pred_res.sort_values('Accuracy Score')      # Order bars in ascending order
g = sns.barplot("Accuracy Score","Algorithm",data = pred_res, order=order['Algorithm'], palette="Set3",orient = "h")

for i in g.patches:         # Put labels on bars
    width = i.get_width()-0.075        # Put labels -0.14 left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Accuracy Score")
g = g.set_title("Accuracy Score by NearMiss")
plt.tight_layout()  
plt.savefig('Accuracy scores before kfold_nm_all models.png')
plt.show()

# Make confusion matrix of each predictor w/o k-fold
matrix =[]  
for cm_pred in y_pred:
    cm = confusion_matrix(y_test, cm_pred)
    matrix.append(cm)
print(matrix)

n=0    # Make the iteration index 0 again so that n doesnt keep getting higher
print('Confusion matrix:')
for p in matrix:
    df_cm = pd.DataFrame(p, index = [i for i in ["On time","Delayed"]],
                  columns = [i for i in ["On time","Delayed"]])
    sns.heatmap(df_cm, center=0.5,
            annot=True, fmt='.0f', cmap='YlGnBu_r',
            vmin=0, vmax=17000)
    plt.title(pred_res['Algorithm'][n])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cm_before kfold_nm_all models ' + str(n) + '.png')
    plt.show()  
    plt.clf()
    n += 1

"""Show the predictive power of the models by cross validating using stratified k fold
"""
kfold = StratifiedKFold(n_splits=5)
cross_val_results_nm = []      # Returns n-fold results of cross validation of each predictor
for classifier in clf:
    cross_val_results_nm.append(cross_val_score(classifier, nm_x_train, nm_y_train, scoring='accuracy', cv=kfold))

cv_means_nm = []    # Returns the means of the n-fold cross val results
cv_std_nm = []      # Returns the standard deviation of the n-fold cross val results
for cv in cross_val_results_nm:
    cv_means_nm.append(cv.mean())
    cv_std_nm.append(cv.std())   

cv_res_nm = pd.DataFrame({"Cross Val Means":cv_means_nm, "Cross Val Errors":cv_std_nm, "Algorithm":["Logistic Regression", "Random Forest", "MLP", "CART", "KNN", "SVM"]})

order = cv_res_nm.sort_values('Cross Val Means')      # Order bars in ascending order
g = sns.barplot("Cross Val Means","Algorithm",data = cv_res_nm, order=order['Algorithm'], palette="Set3",orient = "h", **{'xerr':cv_std_nm})

for i in g.patches:         # Put labels on bars
    width = i.get_width()-(i.get_width()*0.12)        # Put labels left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross Validation Score by NearMiss")
plt.tight_layout()  
plt.savefig('Accuracy scores_kfold_nm_run2.png')
plt.show() 
