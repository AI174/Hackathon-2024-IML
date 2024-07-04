from sklearn.linear_model import LinearRegression
import numpy as np

class Model:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        predictions = self.model.predict(X)
        return np.clip(predictions, 0, None)  # Ensure no negative predictions
