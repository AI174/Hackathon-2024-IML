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

    def preprocess_data2(self, data, is_training=True):
        # Drop rows with missing values
        data.dropna(inplace=True)
        # Convert 'arrival_time' to datetime

        data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S')

        data = data.sort_values(by=['trip_id_unique', 'direction', 'station_index'])
        # Ensure data is sorted by 'trip_id_unique', 'direction', and 'station_index'

        # Group by 'trip_id_unique' and calculate the time difference between the first and last station
        trip_time_diff = data.groupby(['trip_id_unique'])['arrival_time'].agg(
            ['first', 'last'])

        # Step 1: Create a mapping of cluster names to numeric labels
        trip_time_diff['cluster'] = (
            data.groupby('trip_id_unique')['cluster'].agg(['first']).transform(lambda x: pd.factorize(x)[0]))
        # trip_time_diff['cluster'] = .transform(lambda x: pd.factorize(x)[0])

        total_passengers_per_trip = data.groupby('trip_id_unique')['passengers_up'].sum().astype(int)

        station_cor = data.groupby(['trip_id_unique'])[['latitude', 'longitude']].agg(
            ['first', 'last']).astype(float)

        # Calculate trip duration considering potential day change
        trip_time_diff['trip_duration'] = trip_time_diff.apply(
            lambda x: (x['last'] - x['first']).total_seconds() / 60.0
            if x['last'] >= x['first']
            else ((pd.Timedelta(days=1) + x['last'] - x['first']).total_seconds() / 60.0),
            axis=1)

        # Remove rows where trip duration is negative
        trip_time_diff = trip_time_diff[trip_time_diff['trip_duration'] >= 0]

        # Extract the start hour and minute for each trip
        trip_time_diff['start_hour'] = trip_time_diff['first'].dt.hour.astype(int)

        # Merge trip_time_diff back into original data based on 'trip_id_unique'
        trip_time_diff = pd.merge(total_passengers_per_trip,
                                  trip_time_diff,
                                  left_on='trip_id_unique', right_index=True, how='left')




