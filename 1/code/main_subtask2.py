from argparse import ArgumentParser
import pandas as pd
from hackathon_code.data_preprocessor import Preprocessor
import numpy as np
import logging
from sklearn.tree import DecisionTreeRegressor


### running commands:
# --training_set train_bus_schedule.csv --test_set X_trip_duration.csv --out trip_duration_predictions.csv

class DecisionTreeRegressModel:
    def __init__(self):
        self.model = DecisionTreeRegressor(max_depth=12)

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)

def save_predictions(predictions, output_path, col, cols_names):
    # Ensure consistent lengths before saving
    if len(predictions) != len(col):
        min_length = min(len(predictions), len(col))
        predictions = predictions[:min_length]
        col = col[:min_length]

    # Round predictions to the nearest whole number
    rounded_predictions = np.round(predictions).astype(int)
    output = pd.DataFrame({cols_names[0]: col,
                           cols_names[1]: rounded_predictions})
    output.to_csv(output_path, index=False)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--training_set', type=str, required=True,
                        help="path to the training set")
    parser.add_argument('--test_set', type=str, required=True,
                        help="path to the test set")
    parser.add_argument('--out', type=str, required=True,
                        help="path of the output file as required in the task description")
    args = parser.parse_args()

    # 1. load the training set (args.training_set)
    train_data = pd.read_csv(args.training_set, encoding='ISO-8859-1')

    # 2. preprocess the training set
    logging.info("preprocessing train...")
    preprocessor = Preprocessor()
    X_train, y_train, trip_id_unique_train = \
        preprocessor.preprocess_data2(train_data, is_training=True)

    # 3. train a model
    logging.info("training...")
    model = DecisionTreeRegressModel()
    model = model.train(X_train, y_train)  # Corrected: Do not reassign `model`

    # 4. load the test set (args.test_set)
    test_data = pd.read_csv(args.test_set, encoding='ISO-8859-1')

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test, trip_id_unique_test = preprocessor.preprocess_data2(
        test_data, is_training=False)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(
        X_test.drop(columns=['trip_duration_in_minutes']))

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    cols_names = ['trip_id_unique', 'trip_duration_in_minutes']
    save_predictions(predictions, args.out, trip_id_unique_test, cols_names)


