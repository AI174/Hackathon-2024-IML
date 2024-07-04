from argparse import ArgumentParser
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
import numpy as np
import os

def load_data(file_path):
    return pd.read_csv(file_path, encoding='ISO-8859-1')

def inspect_data(data):
    print(data.columns)
    print(data.dtypes)
    print(data.head())

def preprocess_data(data, is_training=True):
    print(f"Initial data shape: {data.shape}")

    # Separate trip_id_unique_station to handle it independently
    trip_id_unique_station = data['trip_id_unique_station'].copy()
    data.drop(columns=['trip_id_unique_station'], inplace=True)

    # Drop specified columns
    columns_to_drop = ['line_id', 'part', 'trip_id_unique', 'station_id', 'station_name']
    data.drop(columns=columns_to_drop, inplace=True)

    # Modify specified columns
    data['direction'] = LabelEncoder().fit_transform(data['direction'])
    data['cluster'] = LabelEncoder().fit_transform(data['cluster'])

    data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
    data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce')

    # Remove rows where arrival_time is null
    data = data[data['arrival_time'].notnull()]

    # Create time_in_station for valid rows
    valid_rows = (data['door_closing_time'].notnull()) & (data['door_closing_time'] >= data['arrival_time']) & (data['door_closing_time'].dt.date == data['arrival_time'].dt.date)
    data['time_in_station'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds()

    # Handle potential negative values in time_in_station
    data.loc[data['time_in_station'] < 0, 'time_in_station'] = np.nan

    average_time_in_station = data['time_in_station'].dropna().mean()

    # Fill time_in_station for invalid rows with the average
    data['time_in_station'].fillna(average_time_in_station, inplace=True)
    data.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

    data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)

    if is_training:
        data['bus_capacity_at_arrival'] = data['passengers_continue'] + data['passengers_up']
    else:
        data['bus_capacity_at_arrival'] = data['passengers_continue']  # No passengers_up in test set

    data.drop(columns=['passengers_continue'], inplace=True)

    # Replace non-numeric values with NaN
    data.replace('#', np.nan, inplace=True)

    print(f"Data shape after dropping columns and replacing values: {data.shape}")

    # Convert all columns to numeric, forcing errors to NaN
    data = data.apply(pd.to_numeric, errors='coerce')

    print(f"Data shape after converting to numeric: {data.shape}")

    # Handle missing values by filling with the median value of the column
    data.fillna(data.median(), inplace=True)

    print(f"Data shape after filling NaN values: {data.shape}")

    # Ensure there are no NaN or infinite values left
    if data.isnull().values.any() or np.isinf(data.values).any():
        print("Warning: NaN or infinity values found in the data after filling NaN values.")
        data = data.dropna()  # Drop rows with remaining NaN or infinity values
        trip_id_unique_station = trip_id_unique_station.loc[data.index]

    # Feature Scaling
    numerical_features = ['station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                          'passengers_continue_menupach', 'time_in_station', 'bus_capacity_at_arrival']
    scaler = StandardScaler()
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    print(f"Data shape after scaling: {data.shape}")
    print(f"Columns after preprocessing: {data.columns}")

    if is_training:
        X = data.drop(columns=['passengers_up'])
        y = data['passengers_up']
        return X, y, trip_id_unique_station
    else:
        return data, trip_id_unique_station

def train_model(X_train, y_train):
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def predict(model, X):
    predictions = model.predict(X)
    return np.clip(predictions, 0, None)  # Ensure no negative predictions

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
    print("Preprocessing training data...")
    X_train, y_train, _ = preprocess_data(train_data, is_training=True)
    print("Preprocessing test data...")
    X_test, test_trip_ids = preprocess_data(test_data, is_training=False)


    # Train the model
    print("Training the model...")
    model = train_model(X_train, y_train)
    # Make predictions
    print("Making predictions...")
    predictions = predict(model, X_test)

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
