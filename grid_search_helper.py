from sklearn.grid_search import GridSearchCV
import numpy as np
import pandas as pd

class EstimatorSelectionHelper:
    '''
    Class to run GridSearchCV simultaneously on different models
    and any hyperparameters you choose for them. Then displays
    the outcome.

    has two methods:

    fit - runs all of the GridSearchCV
    score_summary - displays scores in an ordered pandas dataframe
    '''
    def __init__(self, models, params):
        if not set(models.keys()).issubset(set(params.keys())):
            missing_params = list(set(models.keys()) - set(params.keys()))
            raise ValueError("Some estimators are missing parameters: %s" % missing_params)

        self.models = models
        self.params = params
        self.keys = models.keys()
        self.gridsearches = {}

    def fit(self, X, y, cv=5, pre_dispatch=4, refit=False):
        '''
        Fits all of the models with all of the parameter options
        using cross validation.

        cv = crossvalidation, default is 5
        pre_dispatch = number of jobs run in parallel, default is 4 because
                       my computer has 4 cores
        refit = whether or not it will fit all data to best model from
                crossvalidation, default is False because I don't need
                it so it would waste time
        '''
        for model_name in self.keys:
            print "Running GridSearchCV for {}'s.".format(model_name)
            model = self.models[model_name]
            par = self.params[model_name]

            grid_search = GridSearchCV(model, par, cv=cv, pre_dispatch=pre_dispatch, refit=refit)
            grid_search.fit(X,y)

            self.gridsearches[model_name] = grid_search

    def score_summary(self, sort_by='mean_score'):
        '''
        This builds and prints a pandas dataframe of the summary of all the
        different fits of the models and orders them by best performing
        in a category that you tell it to.
        '''
        def row(key, scores, params):
            d = {'estimator': key,
                 'min_score': np.min(scores),
                 'max_score': np.max(scores),
                 'mean_score': np.mean(scores),
                 'std_score': np.std(scores)
                }
            return pd.Series(dict(params.items() + d.items()))

        rows = []
        for k in self.keys:
            for gsc in self.gridsearches[k].grid_scores_:
                rows.append(row(k, gsc.cv_validation_scores, gsc.parameters))

        df = pd.concat(rows, axis=1).T.sort([sort_by], ascending=False)
        columns = ['estimator', 'min_score', 'mean_score', 'max_score', 'std_score']
        columns = columns + [c for c in df.columns if c not in columns]
        return df[columns]
