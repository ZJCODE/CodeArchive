from sklearn.base import BaseEstimator, TransformerMixin
class Test(BaseEstimator, TransformerMixin):
    def __init__(self, X = True): # no *args or **kargs
        self.X = X
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        # deal with col1 X.ix[:,0]
        # deal with col2 X.ix[:,1]
        # ... ...
        # deal with coln ...
        # combine all cols by using np.c_[]
        # return the result
        return np.c_[X[:,0],[len(i) for i in X[:,1]]]

        

num_pipeline = Pipeline([('test', Test())])