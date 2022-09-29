import pandas as pd
import numpy as np
import os

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler

from chardet.universaldetector import UniversalDetector
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.linear_model import Lasso

#--------------------------------------------------------------------------------------------------#

def undersampledKFold(X_train, y_train):
    kf = StratifiedKFold(n_splits=5)
    r = 1999
    splits = kf.split(X_train, y_train)
    for train_index, test_index in splits:
        rus = RandomUnderSampler(random_state=r)
        X_res_train, y_res_train = rus.fit_resample(train_index.reshape(-1, 1), y_train[train_index])
        X_res_test, y_res_test = rus.fit_resample(test_index.reshape(-1, 1), y_train[test_index])
        r=r+800
        
        yield (np.array(X_res_train.reshape(1, -1)), np.array(X_res_test.reshape(1,-1)))

#--------------------------------------------------------------------------------------------------#

def detect_encoding(file):
    detector = UniversalDetector()
    detector.reset()
    with open(file, 'rb') as f:
        for row in f:
            detector.feed(row)
            if detector.done: break

    detector.close()
    return detector.result['encoding']

#--------------------------------------------------------------------------------------------------#

def standard_scale(train, test):
    xtrain_scaled = pd.DataFrame(StandardScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(StandardScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled

def minmax_scale(train, test):
    xtrain_scaled = pd.DataFrame(MinMaxScaler().fit_transform(train), columns=train.columns)
    xtest_scaled = pd.DataFrame(MinMaxScaler().fit_transform(test), columns=test.columns)
    return xtrain_scaled, xtest_scaled  

#--------------------------------------------------------------------------------------------------#

def normalize_abundances(df): #this is for GEM only
    norm_df = pd.DataFrame()

    for c in df.columns:
        if not c.__contains__('genome_id'):
            total = df.loc[:, c].sum()

            if total == 0: #skip because there is no point in predicting these sites
                continue

            norm_df[c] = df[c] / total

    norm_df['genome_id'] = df['genome_id']
    return norm_df

#--------------------------------------------------------------------------------------------------#

def clean_data(data):
    remove = [col for col in data.columns if data[col].isna().sum() != 0]
    return data.loc[:, ~data.columns.isin(remove)] #this gets rid of remaining NA

#--------------------------------------------------------------------------------------------------#

def split_and_scale_data(features, labels, test_size=0.3):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=5)
    X_train_scaled, X_test_scaled = standard_scale(X_train, X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test

#--------------------------------------------------------------------------------------------------#

def perform_SMOTE(X, y, k_neighbors=5, random_state=1982):
    sm = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_sm, y_sm = sm.fit_resample(X, y)

    return X_sm, y_sm

#--------------------------------------------------------------------------------------------------#

def write_list_to_file(filename, l):
    file = open(filename, 'w')

    for elem in l:
        file.write(elem)
        file.write('\n')

    file.close()

#--------------------------------------------------------------------------------------------------#

def plot_confusion_matrix(y_pred, y_actual, title, path, color=None):
    if color == None:
        color = 'Oranges'

    plt.gca().set_aspect('equal')
    cf_matrix = confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    #print(cf_matrix)

    ax = sns.heatmap(cf_matrix, annot=True, cmap=color, fmt='g')

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['Uncultured','Cultured'])
    ax.yaxis.set_ticklabels(['Uncultured','Cultured'])
    #ax.ticklabel_format(useOffset=False)
    #plt.ticklabel_format(style='plain')

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

def plot_auc(y_pred, y_actual, title, path):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

def plot_feature_importance(columns, importances, path):
    plt.figure(figsize=(16,8))
    sorted_idx = importances.argsort()
    sorted_idx = [i for i in sorted_idx if importances[i] > 0.01]
    plt.barh(columns[sorted_idx], importances[sorted_idx])
    plt.xlabel('Gini Values')
    plt.title('Random Forest Feature Importance')
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

#taken from here: https://stats.stackexchange.com/questions/288736/random-forest-positive-negative-feature-importance
def calculate_pseudo_coefficients(X, y, thr, probs, importances, nfeatures, path):
    dec = list(map(lambda x: (x> thr)*1, probs))
    val_c = X.copy()

    #scale features for visualization
    val_c = pd.DataFrame(StandardScaler().fit_transform(val_c), columns=X.columns)

    val_c = val_c[importances.sort_values('importance', ascending=False).index[0:nfeatures]]
    val_c['t']=y
    val_c['p']=dec
    val_c['err']=np.NAN
    #print(val_c)

    val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'] = 3#'fp'
    val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'] = 2#'tn'
    val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'] = 1#'tp'
    val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'] = 4#'fn'

    n_fp = len(val_c.loc[(val_c['t']==0)&(val_c['p']==1),'err'])
    n_tn = len(val_c.loc[(val_c['t']==0)&(val_c['p']==0),'err'])
    n_tp = len(val_c.loc[(val_c['t']==1)&(val_c['p']==1),'err'])
    n_fn = len(val_c.loc[(val_c['t']==1)&(val_c['p']==0),'err'])

    fp = np.round(val_c[(val_c['t']==0)&(val_c['p']==1)].mean(),2)
    tn = np.round(val_c[(val_c['t']==0)&(val_c['p']==0)].mean(),2)
    tp =  np.round(val_c[(val_c['t']==1)&(val_c['p']==1)].mean(),2)
    fn =  np.round(val_c[(val_c['t']==1)&(val_c['p']==0)].mean(),2)


    c = pd.concat([tp,fp,tn,fn],names=['tp','fp','tn','fn'],axis=1)
    pd.set_option('display.max_colwidth',900)
    c = c[0:-3]

    c.columns = ['TP','FP','TN','FN']

    c.plot.bar()
    plt.title('Relative Scaled Model Coefficients for True/False Positive Rates')
    plt.savefig(path)
    plt.close()

#--------------------------------------------------------------------------------------------------#

def get_rate(act, pred):
    if act == 'Cultured' and pred == 'Cultured':
        return 'TP'
    elif act == 'Cultured' and pred == 'Uncultured':
        return 'FN'
    elif act == 'Uncultured' and pred == 'Cultured':
        return 'FP'
    else:
        return 'TN'

def write_rates_csv(y_actual, y_pred):
    #print(y_actual, y_pred)
    #df = pd.DataFrame(list(zip(y_actual, y_pred)), columns=['Actual', 'Predicted'])
    df = y_actual.copy()
    df.reset_index(inplace=True)
    del df['level_0']
    df.rename(columns={'cultured':'Actual'}, inplace=True)
    df['Predicted'] = y_pred
    #print(df)

    df.replace(1, 'Cultured', inplace=True)
    df.replace(0, 'Uncultured', inplace=True)

    l = [get_rate(elem1, elem2) for elem1, elem2 in list(zip(df['Actual'], df['Predicted']))]

    df['Category'] = l

    #print(df.value_counts())
    df.to_csv('models/LASSO_XGB-classification.csv')
