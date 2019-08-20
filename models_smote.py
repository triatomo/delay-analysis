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
from sklearn.neighbors import NeighborhoodComponentsAnalysis, KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
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

""" 
Fix imbalanced class problem by oversampling the data using SMOTE
"""
# Oversampling under represented classes with SMOTE
print('Oversampling under-represented data...')
sm = SMOTE('not majority', random_state=42)
sm_x_train, sm_y_train = sm.fit_sample(x_train, y_train)

df_y_train = pd.DataFrame(data = sm_y_train, columns= ['Delayed'])
print('Length of oversampled data is', len(sm_x_train))     #62364 
print('Number of delayed shipments in oversampled data is', len(sm_y_train[df_y_train['Delayed']==True]))  #31182
print('Number of on time shipments is', len(sm_y_train[df_y_train['Delayed']==False]))  #31182
print('Proportion of delayed shipments is', len(sm_y_train[df_y_train['Delayed']==True])/len(sm_x_train))
print('Proportion of on time shipments is', len(sm_y_train[df_y_train['Delayed']==False])/len(sm_x_train))

# Scale data to feed prediction models
print('Scaling data...')
scaler = StandardScaler().fit(sm_x_train)

sm_x_train = scaler.transform(sm_x_train)
x_test = scaler.transform(x_test)

"""
Building prediction models
"""
random_state = 1        # Fixate a random state so that the results are reproducible

logreg = LogisticRegression(random_state=random_state)
rf = RandomForestClassifier(random_state=random_state)
mlp = MLPClassifier(hidden_layer_sizes= (50,50,50), max_iter=100)
cart = DecisionTreeClassifier(random_state=random_state)
nca = NeighborhoodComponentsAnalysis(random_state=random_state)
knn = KNeighborsClassifier(n_neighbors=3)
nca_pipe = Pipeline([('nca', nca), ('knn', knn)])           # K nearest Neighbour
svm = SVC(kernel='linear', random_state=random_state)

clf = []        # List of algorithms
clf.append(logreg)
clf.append(rf)
clf.append(mlp)
clf.append(cart)
clf.append(nca_pipe)
clf.append(svm)

"""First the the classifiers will be applied to original test dataset with 16508 instances 
then the same classifiers will be applied to oversampled data
"""
print('Training prediction models on original test dataset (this could take some time)...')
y_pred = []         # List of prediction results
for classifier in clf:
    classifier.fit(sm_x_train, sm_y_train)
    y_pred.append(classifier.predict(x_test))
print(y_pred)

accuracy = []       # List of prediction accuracies by algorithm
for pred in y_pred:
    accuracy.append(accuracy_score(y_test, pred))
print(accuracy)

# df of prediction results above
pred_res = pd.DataFrame({"Accuracy Score":accuracy, "Algorithm":["Logistic Regression", "Random Forest", "MLP"]})

order = pred_res.sort_values('Accuracy Score')      # Order bars in ascending order
g = sns.barplot("Accuracy Score","Algorithm",data = pred_res, order=order['Algorithm'], palette="Set3",orient = "h")

for i in g.patches:         # Put labels on bars
    width = i.get_width()-0.02        # Put labels -0.14 left of the end of the bar i.get_width()/i.get_width()-0.14
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Accuracy Score")
g = g.set_title("Accuracy Score by SMOTE")
plt.tight_layout()  
plt.savefig('Accuracy scores before cross val_sm.png')
plt.show()

# Make confusion matrix of each predictor w/o k-fold
matrix =[]  
for cm_pred in y_pred:
    cm = confusion_matrix(y_test, cm_pred)
    matrix.append(cm)
print(matrix)

n=0     # Make the iteration index 0 again so that n doesnt keep getting higher
print('Confusion matrix:')
for p in matrix:
    df_cm = pd.DataFrame(p, index = [i for i in ["True","False"]],
                  columns = [i for i in ["True","False"]])
    sns.heatmap(df_cm, center=0.5,
            annot=True, fmt='.0f',
            vmin=0, vmax=30830)
    plt.title(pred_res['Algorithm'][n])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cm_before kfold ' + str(n) + '.png')
    plt.show()  
    plt.clf()
    n += 1

# Make confusion matrix of each predictor w/o k-fold
matrix =[]  
for cm_pred in y_pred:
    cm = confusion_matrix(sm_y_test, cm_pred)
    matrix.append(cm)
print(matrix)

m=0     # Make the iteration index 0 again so that n doesnt keep getting higher
print('Confusion matrix:')
for p in matrix:
    df_cm = pd.DataFrame(p, index = [i for i in ["True","False"]],
                  columns = [i for i in ["True","False"]])
    sns.heatmap(df_cm, center=0.5,
            annot=True, fmt='.0f',
            vmin=0, vmax=30830)
    plt.title(pred_res['Algorithm'][m])
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('cm_smotedtest ' + str(m) + '.png')
    plt.show()  
    plt.clf()
    m += 1

# Evaluating NN model and visualization

# for (pred, actual) in zip(y_pred, sm_y_test):   # transform data back into categorical
#     pred = pred.astype(np.int64)
#     pred = le.inverse_transform(pred)
#     actual = actual.astype(np.int64)
#     actual = le.inverse_transform(actual)
# print(y_pred)

# y_pred = y_pred.astype(np.int64)
# y_pred = le.inverse_transform(y_pred)

# sm_y_test = sm_y_test.astype(np.int64)
# sm_y_test = le.inverse_transform(sm_y_test)

matrix =[]  
for cm_pred in y_pred:
    cm = confusion_matrix(sm_y_test, cm_pred)
    matrix.append(cm)
print(matrix)

# c_logreg = confusion_matrix(sm_y_test, y_pred[0])
# print(c_logreg)

print('Confusion matrix:')
for p in matrix:
    df_cm = pd.DataFrame(p, index = [i for i in ["True","False"]],
                  columns = [i for i in ["True","False"]])
    sns.heatmap(df_cm, center=0.5,
            annot=True, fmt='.0f',
            vmin=0, vmax=30830)
    plt.savefig('cm_'+ str(pred_res['Algorithm'])+'.png')       # Only save one file with all algo names
    plt.show()
    plt.clf()


# b = 0
# for (p, b) in matrix: #Doesnt work
#     df_cm = pd.DataFrame(p, index = [i for i in ["True","False"]],
#                   columns = [i for i in ["True","False"]])
#     sns.heatmap(df_cm, center=0.5,
#             annot=True, fmt='.0f',
#             vmin=0, vmax=30830)
#     plt.savefig('cm_'+ str(pred_res['Algorithm'][b])+'.png')
#     plt.show()
#     plt.clf()
#     b+=1

print('Classification matrix:')
print(classification_report(sm_y_test,y_pred)) 

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
    width = i.get_width()-(i.get_width()*0.12)        # Put labels left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross Validation Scores")
plt.tight_layout()  
plt.savefig('Accuracy scores after cross val.png')
plt.show()
