import pandas as pd
import numpy as np
from keras.wrappers.scikit_learn import KerasClassifier
from tensorflow import keras as ks
from sklearn.model_selection import GridSearchCV
from sklearn.manifold import TSNE
from sklearn.linear_model import Lasso
from sklearn.decomposition import PCA

#--------------------------------------------------------------------------------------------------#

class Autoencoder(ks.models.Model):
    def __init__(self, actual_dim, latent_dim, activation, loss, optimizer):
        super(Autoencoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = ks.Sequential([
        ks.layers.Flatten(),
        ks.layers.Dense(latent_dim, activation=activation),
        ])

        self.decoder = ks.Sequential([
        ks.layers.Dense(actual_dim, activation=activation),
        ])

        self.compile(loss=loss, optimizer=optimizer, metrics=[ks.metrics.BinaryAccuracy(name='accuracy')])

    def call(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

def create_AE(actual_dim=1, latent_dim=100, activation='relu', loss='MAE', optimizer='Adam'):
    return Autoencoder(actual_dim, latent_dim, activation, loss, optimizer)

#--------------------------------------------------------------------------------------------------#

def run_PCA(X_train_scaled, X_test_scaled, n_components=0.85):
    pca_model = PCA(n_components=n_components)
    pca_model.fit(X_train_scaled)

    PCA_train = pca_model.transform(X_train_scaled)
    PCA_test = pca_model.transform(X_test_scaled)

    return PCA_train, PCA_test, pca_model

#--------------------------------------------------------------------------------------------------#

def run_LASSO(X_train_scaled, X_test_scaled, y_train, param_grid = None):
    if param_grid == None:
        param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}

    search = GridSearchCV(estimator = Lasso(),
                          param_grid = param_grid,
                          cv = 5,
                          scoring="neg_mean_squared_error",
                          verbose=0
                          )

    search.fit(X_train_scaled, y_train)
    coefficients = search.best_estimator_.coef_
    importance = np.abs(coefficients)
    remove = np.array(X_train_scaled.columns)[importance == 0]

    LASSO_train = X_train_scaled.loc[:, ~X_train_scaled.columns.isin(remove)]
    LASSO_test = X_test_scaled.loc[:, ~X_test_scaled.columns.isin(remove)]

    return LASSO_train, LASSO_test

#--------------------------------------------------------------------------------------------------#

def run_tSNE(X_train_scaled, X_test_scaled, n_components=2):
    tsne = TSNE(n_components=n_components)
    tSNE_train = tsne.fit_transform(X_train_scaled)
    tSNE_test = tsne.fit_transform(X_test_scaled)

    return tSNE_train, tSNE_test

#--------------------------------------------------------------------------------------------------#

def run_UMAP(X_train_scaled, X_test_scaled):
    

#--------------------------------------------------------------------------------------------------#

def run_AE(X_train_scaled, X_test_scaled, param_grid=None):

    if param_grid == None:
        param_grid = {
            'actual_dim' : [len(X_train_scaled.columns)],
            'latent_dim' : [10, 25, 50, 100],
            'activation' : ['relu', 'sigmoid', 'tanh'],
            'loss' : ['MAE', 'binary_crossentropy'],
            'optimizer' : ['SGD', 'Adam']
        }

    model = KerasClassifier(build_fn=create_AE, epochs=10, verbose=0)
    grid = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=5,
        verbose=0
    )

    result = grid.fit(X_train_scaled, X_train_scaled, validation_data=(X_test_scaled, X_test_scaled))
    params = grid.best_params_
    autoencoder = create_AE(**params)

    try:
        encoder_layer = autoencoder.encoder
    except:
        exit

    AE_train = pd.DataFrame(encoder_layer.predict(X_train_scaled))
    AE_train.add_prefix('feature_')
    AE_test = pd.DataFrame(encoder_layer.predict(X_test_scaled))
    AE_test.add_prefix('feature_')

    return AE_train, AE_test
