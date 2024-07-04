from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


class Model:
    def __init__(self):
        self.model = LinearRegression()

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def loss(self, X_val, y_val):
        y_pred = self.predict(X_val)
        mse = mean_squared_error(y_val, y_pred)
        return mse
