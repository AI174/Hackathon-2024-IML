import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


class Preprocessor:
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_data(self, data, is_training=True):

        # Separate trip_id_unique_station to handle it independently
        trip_id_unique_station = data['trip_id_unique_station'].copy()
        data.drop(columns=['trip_id_unique_station'], inplace=True)

        # Drop specified columns
        data.drop(columns=['line_id', 'part', 'trip_id_unique', 'station_id', 'station_name'], inplace=True)

        # convert date
        list_date = ['arrival_time', 'door_closing_time']
        for time in list_date:
            data[time] = pd.to_datetime(data[time], format='%H:%M:%S', errors='coerce')

        # add new col
        arrive_time = data['arrival_time']
        close_time = data['door_closing_time']
        diff = (close_time - arrive_time)
        data['time_in_station'] = diff.dt.total_seconds()

        # convert categorical
        list_categorial = ['direction', 'cluster']
        for categorial in list_categorial:
            data[categorial] = LabelEncoder().fit_transform(data[categorial])

        # Convert true false to numbers
        data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)

        # Remove/Replace all null
        data = data.dropna(subset=['arrival_time'])
        data.replace('#', np.nan, inplace=True)

        # deal with negative values in time_in_station
        data['time_in_station'] = data['time_in_station'].apply(lambda x: np.nan if x < 0 else x)
        average_time_in_station = data['time_in_station'].mean(skipna=True)
        data['time_in_station'] = data['time_in_station'].fillna(average_time_in_station)

        # Remove columns
        data.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

        # Make new column
        if is_training:
            data['bus_capacity_at_arrival'] = data['passengers_continue'] + data['passengers_up']
        else:
            data['bus_capacity_at_arrival'] = data['passengers_continue']

        # Drop the passengers_continue ( no need for it)
        data.drop(columns=['passengers_continue'], inplace=True)

        # Convert all columns to numeric, forcing errors to NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle missing values by filling with the median value of the column
        for col in data.columns:
            median_value = data[col].median()
            data[col].fillna(median_value, inplace=True)

        # Feature Scaling
        data[['station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                              'passengers_continue_menupach', 'time_in_station', 'bus_capacity_at_arrival']] =\
            self.scaler.fit_transform(data[['station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                              'passengers_continue_menupach', 'time_in_station', 'bus_capacity_at_arrival']])
        if is_training:
            X = data.drop(columns=['passengers_up'])
            y = data['passengers_up']
            return X, y, trip_id_unique_station
        else:
            return data, trip_id_unique_station
