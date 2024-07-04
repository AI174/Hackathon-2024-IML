import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve, validation_curve
from data_preprocessor import Preprocessor
from model import Model


class ModelEvaluator:
    def __init__(self, model, X_train, y_train, X_val, y_val):
        self.model = model
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val

    def plot_loss(self):
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)

        train_loss = mean_squared_error(self.y_train, y_train_pred)
        val_loss = mean_squared_error(self.y_val, y_val_pred)

        plt.figure(figsize=(10, 6))
        plt.plot(self.y_train, y_train_pred, 'o', label='Train')
        plt.plot(self.y_val, y_val_pred, 'x', label='Validation')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.title('Predictions vs True Values')
        plt.legend()
        plt.show()

        print(f'Train Loss (MSE): {train_loss}')
        print(f'Validation Loss (MSE): {val_loss}')

    def plot_bias_variance(self):
        y_train_pred = self.model.predict(self.X_train)
        y_val_pred = self.model.predict(self.X_val)

        train_bias = np.mean(y_train_pred - self.y_train)
        val_bias = np.mean(y_val_pred - self.y_val)

        train_variance = np.var(y_train_pred - self.y_train)
        val_variance = np.var(y_val_pred - self.y_val)

        fig, axs = plt.subplots(1, 2, figsize=(14, 6))
        axs[0].bar(['Train Bias', 'Validation Bias'], [train_bias, val_bias])
        axs[0].set_title('Bias')
        axs[1].bar(['Train Variance', 'Validation Variance'], [train_variance, val_variance])
        axs[1].set_title('Variance')
        plt.show()

        print(f'Train Bias: {train_bias}')
        print(f'Validation Bias: {val_bias}')
        print(f'Train Variance: {train_variance}')
        print(f'Validation Variance: {val_variance}')

    def plot_residuals(self):
        y_val_pred = self.model.predict(self.X_val)
        residuals = self.y_val - y_val_pred

        plt.figure(figsize=(10, 6))
        plt.scatter(y_val_pred, residuals)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Values')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.show()

    def plot_prediction_error(self):
        y_val_pred = self.model.predict(self.X_val)

        plt.figure(figsize=(10, 6))
        plt.scatter(self.y_val, y_val_pred)
        plt.plot([self.y_val.min(), self.y_val.max()], [self.y_val.min(), self.y_val.max()], 'r--')
        plt.xlabel('True Values')
        plt.ylabel('Predicted Values')
        plt.title('Prediction Error Plot')
        plt.show()

    def plot_learning_curve(self, train_sizes=np.linspace(0.1, 1.0, 10), cv=5):
        train_sizes, train_scores, val_scores = learning_curve(self.model, self.X_train, self.y_train,
                                                               train_sizes=train_sizes, cv=cv,
                                                               scoring='neg_mean_squared_error')

        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(train_sizes, train_scores_mean, 'o-', label='Training error')
        plt.plot(train_sizes, val_scores_mean, 'x-', label='Validation error')
        plt.xlabel('Training size')
        plt.ylabel('MSE')
        plt.title('Learning Curve')
        plt.legend()
        plt.show()

    def plot_validation_curve(self, param_name, param_range, cv=5):
        train_scores, val_scores = validation_curve(self.model, self.X_train, self.y_train, param_name=param_name,
                                                    param_range=param_range, cv=cv, scoring='neg_mean_squared_error')

        train_scores_mean = -np.mean(train_scores, axis=1)
        val_scores_mean = -np.mean(val_scores, axis=1)

        plt.figure(figsize=(10, 6))
        plt.plot(param_range, train_scores_mean, 'o-', label='Training error')
        plt.plot(param_range, val_scores_mean, 'x-', label='Validation error')
        plt.xlabel(param_name)
        plt.ylabel('MSE')
        plt.title('Validation Curve')
        plt.legend()
        plt.show()


# Example usage:
# Assuming model, X_train, y_train, X_val, y_val are already defined
# model = Model()
# X_train, y_train, _ = preprocessor.preprocess_data(train_data, is_training=True)
# evaluator = ModelEvaluator(model, X_train, y_train, X_val, y_val)
# evaluator.plot_loss()
# evaluator.plot_bias_variance()
# evaluator.plot_residuals()
# evaluator.plot_prediction_error()
# evaluator.plot_learning_curve()
# You need to define the parameter name and range for your specific model for the validation curve
# evaluator.plot_validation_curve(param_name='param_name', param_range=range)
