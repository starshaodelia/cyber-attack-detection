#!/usr/bin/env python
# coding: utf-8

# In[28]:


from sklearn.preprocessing import StandardScaler 
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import *
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, ShuffleSplit, cross_val_score, GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, ConfusionMatrixDisplay, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import shuffle
from sklearn import naive_bayes, tree

from scipy import stats
from string import ascii_letters
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import seaborn as sns
from collections import Counter

from imblearn.combine import SMOTEENN
from imblearn.under_sampling import RandomUnderSampler


# In[2]:


training_set = pd.read_csv("Assignment_2023_training_set_final.csv")
test_set = pd.read_csv("Assignment_2023_test_set_final.csv")


# In[3]:


print("Training set columns: ", len(training_set.columns), "\nTest set columns: ", len(test_set.columns), "\n")
print("Training set rows: ", len(training_set), "\nTest set rows: ", len(test_set), "\n")
test_set_missing_labels = [column for column in training_set.columns if column not in test_set.columns]
print("There are 2 missing columns in the testing set.\nThey are", test_set_missing_labels)
test_values = training_set['attack_cat'].unique()


# In[4]:


unique_columns = training_set["attack_cat"].unique() #gives an array of the unique elements in this column
column_counts = [] # each column, we'll figure out how many instances there are
total = 0 # This is a sanity check to make sure we are looking at all the data
for column in unique_columns:
    print("Searching for ", column, end=": ")
    filtered_data = training_set.query('attack_cat == @column')
    unique_rows =  filtered_data.shape[0]
    print(filtered_data.shape[0], "rows")
    total += unique_rows
    column_counts.append(unique_rows)


# In[5]:


training_set.drop(columns=['ackdat'], inplace=True)
# Plots correlation matrix in the form of a heat map
def correlationHeatMap(df):
    sns.set_theme(style="white")
    correlation = df.corr()
    mask = np.triu(np.ones_like(correlation, dtype=bool))
    f, ax = plt.subplots(figsize=(18,18))
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    corrmat = sns.heatmap(correlation, mask=mask, cmap=cmap, center=0, square=True, linewidths=.5, cbar_kws={'label': 'Correlation', "shrink": .5})
    plt.savefig('corrmat.pdf', dpi='figure', format='pdf')
    # plt.show()

correlationHeatMap(training_set)


# In[6]:


# Drop highly correlated features 
cmat = training_set.corr().abs()
upper = cmat.where(np.triu(np.ones(cmat.shape), k=1).astype(np.bool_)) # obtain upper triangle of cmat
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)] # find features w/ 90% correlation
training_set.drop(to_drop, axis=1, inplace=True)
test_set.drop(to_drop, axis=1, inplace=True)
test_set.drop(columns=['ackdat'], inplace=True)


# In[7]:


# Standardising columns with numeric values
numcols_train = training_set.drop(columns=['id', 'label', 'is_ftp_login'])
numcols_test = test_set.drop(columns=['id'])
numeric_columns_train = numcols_train.select_dtypes(include=['int', 'float']).columns.tolist()
numeric_columns_test = numcols_test.select_dtypes(include=['int', 'float']).columns.tolist()
scale = StandardScaler()
training_set[numeric_columns_train] = scale.fit_transform(training_set[numeric_columns_train])
test_set[numeric_columns_test] = scale.fit_transform(test_set[numeric_columns_test])
#print('Number of columns with numeric data: ', len(training_set[numeric_columns].columns))

# Finding categorical features to label encode 
le = LabelEncoder()
#train_set = training_set.drop('attack_cat_names', axis=1)
categorical_features_train = training_set.select_dtypes(include='object')
categorical_features_test = test_set.select_dtypes(include='object')
for c in categorical_features_train:
    le.fit(categorical_features_train[c])
    training_set[c]=le.transform(categorical_features_train[c])

for c in categorical_features_test:
    le.fit(categorical_features_test[c])
    test_set[c]=le.transform(categorical_features_test[c])


# In[8]:


test_keys = training_set['attack_cat'].unique()
res_dict = {test_keys[i]: test_values[i] for i in range(len(test_keys))}
res = sorted(res_dict.items())
cat = list(zip(*res))[1]


# In[9]:


smote_enn = SMOTEENN(random_state=505)
norm = training_set[training_set["label"] == 0]
mal = training_set[training_set["label"] == 1]

X = training_set.loc[:, training_set.columns != 'attack_cat']
y = training_set['attack_cat']
X_resampled, y_resampled = smote_enn.fit_resample(X, y)
print(sorted(Counter(y_resampled).items()))


# In[10]:


rus = RandomUnderSampler(sampling_strategy='all', random_state=42)
X_res, y_res = rus.fit_resample(X_resampled, y_resampled)
print(sorted(Counter(y_res).items()))


# In[11]:


df = pd.concat([X_res, y_res], axis=1)
df = shuffle(df).reset_index(drop=True)

c = list(df.columns)
a, b = c.index('attack_cat'), c.index('label')
c[b], c[a] = c[a], c[b]
df = df[c]
training_set = df
training_set.to_csv('training.csv')


# In[ ]:





# In[ ]:





# In[12]:


# Now the dataset needs to be separated into two subsets
train = training_set.iloc[:,:-2]
test = training_set['attack_cat']
X_train, X_test, y_train, y_test = train_test_split(train, test,
                                                    test_size=0.1,
                                                    random_state=42)
X_train.shape, X_test.shape


# In[13]:


val = pd.concat([X_test, y_test], axis=1)
val.to_csv('validation.csv')  


# In[14]:


# This is random sampling
ss = ShuffleSplit(n_splits=10, random_state=4)
# This is non-random sampling, we just break the data in to 10 contiguous sub-sets
kf = KFold(n_splits=10)
# Ensuring the balance between classes in the model/validate sets
# means we should use stratified sampling
skf = StratifiedKFold(n_splits=10)


# In[15]:


cmap_data = plt.cm.Paired
cmap_cv = plt.cm.coolwarm

def plot_cv_indices(cv, X, y, group, ax, n_splits, lw=10):
    """
    Create a sample plot for indices of a cross-validation object.
    Adapted from https://scikit-learn.org/stable/auto_examples/model_selection/plot_cv_indices.html#define-a-function-to-visualize-cross-validation-behavior

    Parameters
    ----------
    cv: cross validation method

    X : training data

    y : data labels

    group : group labels

    ax : matplolib axes object

    n_splits : number of splits

    lw : line width for plotting
    """

    # Generate the training/testing visualizations for each CV split
    for ii, (tr, tt) in enumerate(cv.split(X=X, y=y, groups=group)):
        # Fill in indices with the training/test groups
        indices = np.array([np.nan] * len(X))
        indices[tt] = 1
        indices[tr] = 0

        # Visualize the results
        ax.scatter(range(len(indices)), [ii + .5] * len(indices),
                   c=indices, marker='_', lw=lw, cmap=cmap_cv,
                   vmin=-.2, vmax=1.2)

    # Plot the data classes and groups at the end
    ax.scatter(range(len(X)), [ii + 1.5] * len(X),
               c=y, marker='_', lw=lw, cmap=cmap_data)

    #ax.scatter(range(len(X)), [ii + 2.5] * len(X),
    #           c=group, marker='_', lw=lw, cmap=cmap_data)

    # Formatting
    yticklabels = list(range(n_splits)) + ['class', 'group']
    ax.set(yticks=np.arange(n_splits+2) + .5, yticklabels=yticklabels,
           xlabel='Sample index', ylabel="CV iteration",
           ylim=[n_splits+2.2, -.2])
    ax.set_title('{}'.format(type(cv).__name__), fontsize=15)
    return ax


# In[16]:


scoring = ['accuracy','f1_macro']

# Set up a figure with three subplots
fig, ax = plt.subplots(1,3, figsize=(18,6))
# visualise the ShulffleSplit algorithm
plot_cv_indices(ss,
                train, test,
                group=None,
                ax=ax[0],
                n_splits=10)
# visualise the KFolds algorithm
plot_cv_indices(kf,
                train, test,
                group=None,
                ax=ax[1],
                n_splits=10)
# visualise the StratifiedKFolds algorithm
plot_cv_indices(skf,
                train, test,
                group=None,
                ax=ax[2],
                n_splits=10)
plt.savefig('crossval.pdf', dpi='figure', format='pdf')
plt.show()


# In[17]:


parameters = {'weights': ['uniform','distance'], 
              'n_neighbors':[110,120,130]} 
knn = KNeighborsClassifier()
gscv = GridSearchCV(estimator=knn,
                    param_grid=parameters,
                    cv=kf,  # the cross validation folding pattern
                    scoring=scoring, refit='accuracy', return_train_score=True)
# model training: 
best_knn = gscv.fit(X_train, y_train)


# In[18]:


best_knn.best_params_, best_knn.best_score_


# In[19]:


knn = KNeighborsClassifier(weights = best_knn.best_params_['weights'],
                            n_neighbors = best_knn.best_params_['n_neighbors'])
knn.fit(X_train, y_train)


# In[20]:


fig, ax = plt.subplots(1,1, figsize=(6, 6))

ConfusionMatrixDisplay.from_estimator(knn,
                                      X_test, y_test,
                                      display_labels=cat,
                                      ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('knnmat.pdf', dpi='figure', format='pdf')
plt.show()


# In[40]:


testRes = pd.DataFrame(knn.predict(test_set), columns=['knn_num'])


# In[41]:


testRes['id'] = test_set['id']
testRes = testRes[['id', 'knn_num']]
testRes['Predict1'] = testRes['knn_num'].map(res_dict)


# In[30]:


parameters = {'criterion': ('gini','entropy'), 
              'min_samples_split':[5000, 7500, 10000, 20000]}
dtc = tree.DecisionTreeClassifier()
gscv = GridSearchCV(estimator=dtc,
                    param_grid=parameters,
                    cv=kf,
                    scoring='accuracy')
best_dtc = gscv.fit(X_train, y_train)
best_dtc.best_params_, best_dtc.best_score_


# In[31]:


dtc = tree.DecisionTreeClassifier(criterion=best_dtc.best_params_['criterion'],
                                  min_samples_split=best_dtc.best_params_['min_samples_split'])
dtc.fit(X_train, y_train)


# In[32]:


fig, ax = plt.subplots(1,1, figsize=(90,45))
tree.plot_tree(dtc,
               filled=True,
               ax=ax, fontsize=18)
plt.savefig('dtc.pdf', dpi='figure', format='pdf')
plt.show()


# In[33]:


fig, ax = plt.subplots(1,1, figsize=(6, 6))

ConfusionMatrixDisplay.from_estimator(dtc,
                                      X_test, y_test,
                                      display_labels=cat,
                                      ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('dtcmat.pdf', dpi='figure', format='pdf')
plt.show()


# In[42]:


testRes['dtc_num'] = dtc.predict(test_set)
testRes['Predict2'] = testRes['dtc_num'].map(res_dict)
testRes.drop(['knn_num', 'dtc_num'], axis=1, inplace=True)
testRes.to_csv('predictions.csv')


# In[ ]:


fig, ax = plt.subplots(1,1)
nb = naive_bayes.GaussianNB()
nb.fit(X_train, y_train)
y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"NB accuracy is {accuracy:5.3f}")

ConfusionMatrixDisplay.from_estimator(nb,
                                      X_test, y_test,
                                      display_labels=cat,
                                      ax=ax)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('nb.pdf', dpi='figure', format='pdf')
plt.show()


# In[ ]:


y_pred_knn = knn.predict(X_test)
repKNN = classification_report(y_test, y_pred_knn, output_dict=True)
dfKNN = pd.DataFrame(repKNN).transpose()
print(dfKNN.to_latex())


# In[ ]:


y_pred_dtc = dtc.predict(X_test)
repDTC = classification_report(y_test, y_pred_dtc, output_dict=True)
dfDTC = pd.DataFrame(repDTC).transpose()
print(dfDTC.to_latex())


# In[ ]:


y_pred_nb = nb.predict(X_test)
repNB = classification_report(y_test, y_pred_nb, output_dict=True)
dfNB = pd.DataFrame(repNB).transpose()
print(dfNB.to_latex())

