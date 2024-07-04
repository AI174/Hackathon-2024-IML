import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, data, is_training=True):
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
        data[numerical_features] = self.scaler.fit_transform(data[numerical_features])

        print(f"Data shape after scaling: {data.shape}")
        print(f"Columns after preprocessing: {data.columns}")

        if is_training:
            X = data.drop(columns=['passengers_up'])
            y = data['passengers_up']
            return X, y, trip_id_unique_station
        else:
            return data, trip_id_unique_station
