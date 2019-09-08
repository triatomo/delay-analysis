import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LogisticRegression, LassoCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier      # NeighborhoodComponentsAnalysis cannot import
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, learning_curve
from imblearn.pipeline import make_pipeline, Pipeline

df = pd.read_csv('cleaned_data.csv', dtype={'Delayed': np.bool})
pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 4)

"""
Split data into training and test dataset
"""
# Define columns relevant for the prediction
model_vars = ['Customer Address Country', 'Carrier', 'PPU day', 'FHS day', 'FDA day', 'Delivery day', 'PPU-FHS', 'FHS-FDA', 'FDA-Delivery', 'Delayed']

rel_data = df[model_vars]
rel_data_encoded = pd.get_dummies(rel_data)     # convert categorical vars into numerical.Yields 56 cols

# Separate predictor from target variable
x = rel_data_encoded.drop(['Delayed'], axis = 1)       # predictor vars 55 x 50024
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
# sm_x, sm_y = sm.fit_sample(x_scale, y)

# df_y_train = pd.DataFrame(data = sm_y, columns= ['Delayed'])
# print('Length of oversampled data is', len(sm_x))     #62364 
# print('Number of delayed shipments in oversampled data is', len(sm_y[df_y_train['Delayed']==True]))  #31182
# print('Number of on time shipments is', len(sm_y[df_y_train['Delayed']==False]))  #31182

"""
Building prediction models
"""
random_state = 1        # Fixate a random state so that the results are reproducible
n_jobs = -1

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
    cv_score = cross_val_score(smoted_classifier, x_scale, y, scoring='accuracy', cv=kfold, n_jobs=-1)
    cross_val_results_sm.append(cv_score)

cv_means_sm = []    # Returns the means of the n-fold cross val results
cv_std_sm = []      # Returns the standard deviation of the n-fold cross val results
for cv in cross_val_results_sm:
    cv_means_sm.append(cv.mean())
    cv_std_sm.append(cv.std())   

cv_res_sm = pd.DataFrame({"Cross Val Means":cv_means_sm, "Cross Val Errors":cv_std_sm, "Algorithm":["Logistic Regression", "Random Forest", "MLP", "CART", "KNN", "SVM"]})

order = cv_res_sm.sort_values('Cross Val Means')      # Order bars in ascending order
g = sns.barplot("Cross Val Means","Algorithm",data = cv_res_sm, order=order['Algorithm'], palette="Set3",orient = "h", **{'xerr':cv_std_sm})

for i in g.patches:         # Put labels on bars
    width = i.get_width()-(i.get_width()*0.12)        # Put labels left of the end of the bar
    g.text(width, i.get_y() + i.get_height()/2, round(i.get_width(),3), color='black', va="center")
    
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross Validation Score by SMOTE")
plt.tight_layout()  
plt.savefig('Accuracy scores_kfold_sm.png')
plt.show()

"""
Hyperparameter tuning for kNN, MLP, Random forest and CART (SMOTE)
"""
random_state = 1
kfold = StratifiedKFold(n_splits=5)
sm = SMOTE('not majority', random_state=42)

# Random Forest
rf = RandomForestClassifier(random_state=random_state)

sm_classifier_rf = Pipeline([('sm', sm), ('rf', rf)])
param_grid = {"rf__max_features": [1, 3, 10],
              "rf__min_samples_split": [2, 3, 10],
              "rf__min_samples_leaf": [1, 3, 10],
              "rf__n_estimators" :[100,300]}

gs_rf = GridSearchCV(sm_classifier_rf, param_grid = param_grid, cv=kfold, scoring="accuracy", verbose = 1, n_jobs = -1)
gs_rf.fit(x_scale, y)

rf_best = gs_rf.best_estimator_.named_steps['rf']

print('Best parameters found: ', gs_rf.best_params_)
print('Best estimator found: ', rf_best)
print('Best score found: ', gs_rf.best_score_)

# KNN
knn = KNeighborsClassifier()

sm_classifier_knn = Pipeline([('sm', sm), ('knn', knn)])
param_grid_knn = {"knn__n_neighbors": [3, 5, 11, 19],
                  "knn__weights": ['uniform', 'distance']}

gs_knn = GridSearchCV(sm_classifier_knn, param_grid=param_grid_knn, cv=kfold, scoring="accuracy", verbose = 1, n_jobs = -1)
gs_knn.fit(x_scale, y)

knn_best = gs_knn.best_estimator_.named_steps['knn']

print('Best parameters found: ', gs_knn.best_params_)
print('Best estimator found: ', knn_best)
print('Best score found: ', gs_knn.best_score_)

# CART
cart = DecisionTreeClassifier(random_state=random_state)

sm_classifier_cart = Pipeline([('sm', sm), ('cart', cart)])
param_grid_cart = {'cart__min_samples_split':[2,3,10],
                   'cart__max_depth':[1,20,2],
                   'cart__min_samples_leaf': [1,3,10]}

# Message from Ben: use n_jobs = -1 to make it run on all cores in parallel (much faster)
gs_cart = GridSearchCV(sm_classifier_cart, param_grid=param_grid_cart, cv=kfold, scoring="accuracy", verbose = 1, n_jobs = -1)
gs_cart.fit(x_scale, y)

cart_best = gs_cart.best_estimator_.named_steps['cart']

print('Best parameters found: ', gs_cart.best_params_)
print('Best estimator found: ', cart_best)
print('Best score found: ', gs_cart.best_score_)

""" Generate a simple plot of the test and training learning curve
"""
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=-1, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.savefig(title + '.png')
    return plt

g = plot_learning_curve(gs_knn.best_estimator_,"kNN learning curves", x_scale, y,cv=kfold)
g = plot_learning_curve(gs_cart.best_estimator_,"CART learning curves", x_scale, y,cv=kfold)
g = plot_learning_curve(gs_rf.best_estimator_,"Random Forest learning curves", x_scale, y,cv=kfold)

""" Feature Importance
"""
nrows = ncols = 0
fig, axes = plt.subplots(nrows = nrows, ncols = ncols, sharex="all", figsize=(15,15))

names_classifiers = [("CART",cart_best), ("Random Forest",rf_best)]

nclassifier = 0
for i in names_classifiers:
        name = i[0]
        classifier = i[1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:25] 
        g = sns.barplot(y=x.columns[indices], x = classifier.feature_importances_[indices], orient='h')
        for s in g.patches:
            width = s.get_width()       # Put labels left of the end of the bar
            g.text(width, s.get_y() + s.get_height()/2, round(s.get_width(),3), color='black', va="center")
        # nested for loop doesnt work properly. RF has two label on each bar
        g.set_xlabel("Relative importance",fontsize=12)
        g.set_ylabel("Features",fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        plt.savefig(name + ' feature importance.png', bbox_inches='tight')

# Feature importance for kNN

random_state = 1
kfold = StratifiedKFold(n_splits=5)

col = x.columns
df_x_scale = pd.DataFrame(x_scale, columns = col)       # Should use x_scale. x yields the wrong importance

reg = LassoCV(cv=kfold, random_state=random_state, n_jobs=-1)
reg.fit(df_x_scale, y)
print("Best alpha using built-in LassoCV: %f" % reg.alpha_)
print("Best score using built-in LassoCV: %f" %reg.score(df_x_scale,y))
coef = pd.Series(reg.coef_, index = x.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " +  str(sum(coef == 0)) + " variables")

nonzero_coef = coef[coef != 0] 
imp_coef = nonzero_coef.sort_values()
plt.rcParams['figure.figsize'] = (15, 15.0)
g = imp_coef.plot(kind = "barh")
for (s, i) in zip(g.patches, imp_coef):
    width = s.get_width()       # Put labels left of the end of the bar
    if i > 0:
        g.text(width, s.get_y() + s.get_height()/2, round(s.get_width(),3), color='black', va="center")
    else:
        g.text(width-0.006, s.get_y() + s.get_height()/2, round(s.get_width(),3), color='black', va="center")
plt.title("Feature importance using Lasso Model")
g.set_xlabel("Relative importance",fontsize=12)
g.set_ylabel("Features",fontsize=12)
g.tick_params(labelsize=9)
plt.tight_layout()  
plt.savefig('Feature importance Lasso_knn.png', bbox_inches='tight')
plt.show()