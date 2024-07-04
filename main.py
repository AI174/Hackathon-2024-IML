from argparse import ArgumentParser
import pandas as pd
from data_preprocessor import Preprocessor
from model import LinearRegressModel
import numpy as np
import logging


def save_predictions(predictions, output_path, trip_id_unique_station):
    # Ensure consistent lengths before saving
    if len(predictions) != len(trip_id_unique_station):
        min_length = min(len(predictions), len(trip_id_unique_station))
        predictions = predictions[:min_length]
        trip_id_unique_station = trip_id_unique_station[:min_length]

    # Round predictions to the nearest whole number
    rounded_predictions = np.round(predictions).astype(int)
    output = pd.DataFrame({'trip_id_unique_station': trip_id_unique_station, 'passengers_up': rounded_predictions})
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
    X_train, y_train, trip_id_unique_station_train = preprocessor.preprocess_data(train_data, is_training=True)

    # 3. train a model
    logging.info("training...")
    model = LinearRegressModel()
    model.train(X_train, y_train)  # Corrected: Do not reassign `model`

    # 4. load the test set (args.test_set)
    test_data = pd.read_csv(args.test_set, encoding='ISO-8859-1')

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test, trip_id_unique_station_test = preprocessor.preprocess_data(test_data, is_training=False)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(X_test)

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    save_predictions(predictions, args.out, trip_id_unique_station_test)
