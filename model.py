from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
import numpy as np


class LinearRegressModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X):
        predictions = self.model.predict(X)
        # Ensure no negative predictions
        predictions[predictions < 0] = 0
        return predictions


class DecisionTreeRegressModel:
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=12)   # I don't know wtf is this number

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
