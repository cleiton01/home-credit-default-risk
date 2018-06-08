import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import RadiusNeighborsRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPRegressor
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

"""
    data_set is the set that will be used, only with the column used on the model.
    targe is the value that we try to predict. 
"""
#X_train, X_test, y_train, y_test = train_test_split(data_set, target, test_size=0.4, random_state=0)

class Regression():

    def get_kernels(key):
        kernels = {1:'linear', 2:'poly', 3:'rbf', 4:'sigmoid'}

        return kernels.get(key)

    def get_loss(key):
        loss = {1:'squared_loss', 2:'huber', 3:'epsilon_insensitive', 4:'squared_epsilon_insensitive'}

        return loss.get(key)

    def get_penalty(key):
        penalty={1:None, 2:'l2', 3:'l1', 4:'elasticnet'}

        return penalty.get(key)

    def get_weights(key):
        weights = {1:'uniform', 2:'distance'}
        
        return weights.get(key)

    def algorithm_KRN(key):

        algorithm = {1:'ball_tree', 2:'kd_tree', 3:'brute'}

        return algorithm.get(key)


    def Power_parameter(key):

        p = {1:1, 2:2}

        return p.get(key)

    def get_criterion(key):

        criterion = {1:'mse',2:'friedman_mse', 3:'mae'}

        return criterion.get(key)


        """
        Support Vector Machines

        gamma usar valores baixos 0.xxxx
        degree apenas para o kernel poly eh um valor inteiro
        """
    def svr(self, kernel='linear', epsilon=0.1, degree=3, gamma='auto', cache_size=50):

        model = svm.SVR(kernel=kernel, C=1, epsilon=epsilon, degree=degree, gamma=gamma, cache_size=cache_size)

        return model

    """
    Stochastic Gradient Descent
    """
    def SGDRegressor(loss='squared_loss', penalty='l2', epoch=10):

        model = linear_model.SGDRegressor(loss=loss, penalty=penalty, alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=epoch, tol=None, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False, n_iter=None)
        """
            y = np.random.randn(n_samples)
            X = np.random.randn(n_samples, n_features)
            clf = linear_model.SGDRegressor()
            clf.fit(X, y)
        """
        return model

    """
        Nearest Neighbors
    """
    def KNeighborsRegressor(neighbors=5, weights='distance', algorithm='auto', p=2):

        model = KNeighborsRegressor(n_neighbors=neighbors, weights=weights, algorithm=algorithm, leaf_size=30, p=p, metric='minkowski', metric_params=None, n_jobs=1)

        return model

    """
        Nearest Neighbors
    """
    def RadiusNeighborsRegressor(radius=1.0, weights='distance', algorithm='auto', p=2):

        model = RadiusNeighborsRegressor(radius=radius, weights=weights, algorithm=algorithm, leaf_size=30, p=p, metric='minkowski', metric_params=None)

        return model

    """
        Decision Trees
    """
    def DecisionTreeRegressor(criterion='mse', splitter='best'):

        model = tree.DecisionTreeRegressor(criterion=criterion, splitter=splitter, max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features=None, random_state=None, max_leaf_nodes=None, min_impurity_decrease=0.0, presort=False)

        return model

    def loss_GRADIENT(key):
        loss = {1:'ls', 2:'lad', 3:'huber', 4:'quantile'}

        return loss.get(key)
    
    """
        Ensemble methods
    """

    def GradientBoostingRegressor(loss=0.1,learning_rate=0.1, n_estimators=100, max_depth=3, criterion='friedman_mse'):

        model = GradientBoostingRegressor(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

        return model

    def get_loss_ADAB(key):
        
        loss = {'linear', 'square', 'exponential'}

        return loss.get(key)

    def AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear', random_state=None):

        model = AdaBoostRegressor(base_estimator=base_estimator, n_estimators=n_estimators, learning_rate=learning_rate, loss=loss, random_state=random_state)

        return model


    def LogisticRegression(random_state=1):

        model = LogisticRegression(random_state=random_state)

        return model

    """
        Multiclass and multilabel algorithms
    """

    def MultiOutputRegressor(model):
        from sklearn.multioutput import MultiOutputRegressor
        mult_model = MultiOutputRegressor(model)
        mult_model.fit(X,Y).predict(x_predict)

    """
        Multiclass and multilabel algorithms
    """
    def MultiOutputClassifier(model):
        from sklearn.multioutput import MultiOutputClassifier
        mult_model = MultiOutputClassifier(model, n_jobs=-1)
        mult_model.fit(X, Y).predict(x_predict)

    """
        Neural network models (supervised)
    """
    def MLPRegressor(hidden_layer_sizes=(100, ), activation='relu', solver='adam', alpha=0.0001, batch_size='auto', learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200):

        model = MLPRegressor(hidden_layer_sizes=hidden_layer_sizes, activation=activation, solver=solver, alpha=alpha, batch_size=batch_size, learning_rate=learning_rate, learning_rate_init=learning_rate_init, power_t=power_t, max_iter=max_iter)

        return model

    def Sequential(dim=(10, 10, 10, 1), kernel_initializer='normal', activation='relu', loss='mean_square_error', optimizer='adam'):
        model = Sequential()
        model.add(Danse(dim[0], input_dim=dim[0], kernel_initializer=kernel_initializer, activation=activation))
        model.add(Dense(dim[1], kernel_initializer=kernel_initializer, activation=activation))
        model.add(Dense(dim[2], kernel_initializer=kernel_initializer, activation=activation))
        model.add(Dense(dim[3], kernel_initializer=kernel_initializer))

        model.compile(loss=loss, optimizer=optimizer)

        return model
    """
    Ensemble methods
        GradientBoostingRegressor
        AdaBoostRegressor
    Neural network models (supervised)
    """