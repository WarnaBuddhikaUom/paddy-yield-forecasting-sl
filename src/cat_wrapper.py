"""Thin scikit-learn wrapper around CatBoostRegressor.

Recent scikit-learn versions refuse to clone CatBoostRegressor directly (it copies `cat_features`
inside get_params), which breaks RandomizedSearchCV and StackingRegressor. This wrapper stores plain
hyper-parameters and builds the CatBoost model inside fit().
"""
from sklearn.base import BaseEstimator, RegressorMixin
from catboost import CatBoostRegressor


class CatBoostSK(RegressorMixin, BaseEstimator):
    def __init__(self, iterations=800, learning_rate=0.05, depth=6, l2_leaf_reg=3.0,
                 bagging_temperature=1.0, cat_cols=("District", "Season"), random_state=42, thread_count=2):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.bagging_temperature = bagging_temperature
        self.cat_cols = cat_cols
        self.random_state = random_state
        self.thread_count = thread_count

    def fit(self, X, y):
        self.model_ = CatBoostRegressor(
            iterations=self.iterations, learning_rate=self.learning_rate, depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg, bagging_temperature=self.bagging_temperature,
            cat_features=list(self.cat_cols), random_seed=self.random_state,
            thread_count=self.thread_count, verbose=0, allow_writing_files=False)
        self.model_.fit(X, y)
        self.feature_names_in_ = list(X.columns)
        return self

    def predict(self, X):
        return self.model_.predict(X)
