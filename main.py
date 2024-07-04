from argparse import ArgumentParser
import pandas as pd
from data_preprocessor import Preprocessor
from model import Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

def inspect_data(data):
    print(data.columns)
    print(data.dtypes)
    print(data.head())

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

def main(train_file, test_file, output_file):
    # Load the data
    print("Loading data...")
    train_data = load_data(train_file)
    test_data = load_data(test_file)

    # Inspect the data
    print("Inspecting training data...")
    inspect_data(train_data)
    print("Inspecting test data...")
    inspect_data(test_data)

    # Preprocess the data
    preprocessor = Preprocessor()
    print("Preprocessing training data...")
    X_train, y_train, _ = preprocessor.preprocess_data(train_data, is_training=True)
    print("Preprocessing test data...")
    X_test, test_trip_ids = preprocessor.preprocess_data(test_data, is_training=False)

    # Train the model and calculate MSE for different percentages of training data
    percentages = [10, 20, 40, 60, 80, 100]
    mse_scores = []

    for percentage in percentages:
        print(f"Training model with {percentage}% of the data...")
        if percentage < 100:
            partial_X_train, _, partial_y_train, _ = train_test_split(X_train, y_train, train_size=percentage / 100, random_state=42)
        else:
            partial_X_train = X_train
            partial_y_train = y_train

        model = Model()
        model.train(partial_X_train, partial_y_train)
        predictions = model.predict(X_train)
        mse = mean_squared_error(y_train, predictions)
        mse_scores.append(mse)
        print(f"MSE with {percentage}% of the data: {mse}")

    print(f"Mean Squared Error at different training sizes: {mse_scores}")

    # Train the final model on the entire training set
    print("Training final model on the entire training set...")
    final_model = Model()
    final_model.train(X_train, y_train)

    # Make predictions
    print("Making predictions on the test set...")
    predictions = final_model.predict(X_test)

    # Save predictions
    print(f"Saving predictions to {output_file}...")
    save_predictions(predictions, output_file, test_trip_ids)
    print("Predictions saved successfully.")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train_file", type=str, required=True, help="Path to the training data file.")
    parser.add_argument("--test_file", type=str, required=True, help="Path to the test data file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save the output predictions.")
    args = parser.parse_args()

    main(args.train_file, args.test_file, args.output_file)
