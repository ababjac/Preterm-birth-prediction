import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.metrics import roc_curve, auc, confusion_matrix
import numpy as np

#--------------------------------------------------------------------------------------------------#

def plot_points(y_pred, y_actual, title):
    plt.scatter(y_actual, y_pred)
    plt.title(title)
    plt.xlabel('Delivery Week')
    plt.ylabel('Predicted Delivery')
    
    plt.show()
    plt.clf()

#--------------------------------------------------------------------------------------------------#

def plot_confusion_matrix(y_pred, y_actual, title, path=None, color=None):
    if color == None:
        color = 'Oranges'

    plt.gca().set_aspect('equal')
    cf_matrix = confusion_matrix(y_actual, y_pred)
    if len(cf_matrix) != 2: #if it predicts perfectly then confusion matrix returns incorrect form
        val = cf_matrix[0][0]
        tmp = [val, 0]
        cf_matrix = np.array([tmp, [0, 0]])

    ax = sns.heatmap(cf_matrix, annot=True, cmap=color)

    ax.set_title(title+'\n\n');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])

    ## Display the visualization of the Confusion Matrix.
    plt.tight_layout()
    #plt.savefig(path)
    plt.show()
    plt.clf()

#--------------------------------------------------------------------------------------------------#

def plot_auc(y_pred, y_actual, title, path=None):
    fpr, tpr, thresholds = roc_curve(y_actual, y_pred)
    roc_auc = auc(fpr, tpr)
    print('AUC:', roc_auc)
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)

    plt.title(title)
    plt.legend()
    #plt.savefig(path)
    plt.show()
    plt.clf()

#--------------------------------------------------------------------------------------------------#

def plot_pca(colors, pca, components, path=None):

    labels = {
        str(i): f"PC {i+1} ({var:.1f}%)"
        for i, var in enumerate(pca.explained_variance_ratio_ * 100)
    }

    #print(labels)
    fig = px.scatter_matrix(
        components,
        labels=labels,
        dimensions=range(9),
        color=colors
    )

    fig.update_traces(diagonal_visible=False)
    #fig.write_image(path)
    fig.show()
    #fig.clf()
