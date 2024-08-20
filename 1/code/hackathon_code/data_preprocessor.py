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

        # Drop unnecessary columns
        drop_columns = ['part', 'station_id',
                        'trip_id_unique', 'station_name']
        data.drop(columns=drop_columns, inplace=True)

        # Convert time columns to datetime
        time_columns = ['arrival_time', 'door_closing_time']
        data[time_columns] = data[time_columns].apply(pd.to_datetime,
                                                      format='%H:%M:%S',
                                                      errors='coerce')

        # Calculate 'time_in_station'
        data['time_in_station'] = (data['door_closing_time'] - data[
            'arrival_time']).dt.total_seconds()

        # Encode categorical variables
        categorical_columns = ['direction', 'cluster']
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        for col in categorical_columns:
            data[col] = label_encoders[col].fit_transform(data[col])

        # Convert 'arrival_is_estimated' to int
        data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)

        # Drop rows with null 'arrival_time'
        data.dropna(subset=['arrival_time'], inplace=True)

        # Replace '#' with NaN
        data.replace('#', np.nan, inplace=True)

        # Handle negative values in 'time_in_station'
        data.loc[data[
                     'time_in_station'] < 0, 'time_in_station'] = np.nan  # Changed to use .loc
        average_time_in_station = data['time_in_station'].mean(skipna=True)
        data['time_in_station'].fillna(average_time_in_station,
                                       inplace=True)  # Changed to use .fillna

        # Drop 'arrival_time' and 'door_closing_time' columns
        data.drop(columns=['door_closing_time'], inplace=True)

        # Calculate 'bus_capacity_at_arrival'
        if is_training:
            data['bus_capacity_at_arrival'] = data['passengers_continue'] + \
                                              data['passengers_up']
        else:
            data['bus_capacity_at_arrival'] = data['passengers_continue']

        # Drop 'passengers_continue'
        data.drop(columns=['passengers_continue'], inplace=True)

        # Convert all columns to numeric, forcing errors to NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col],
                                      errors='coerce')  # Changed to loop for conversion

        # Handle missing values by filling with the median value of the column
        for col in data.columns:
            data[col].fillna(data[col].median(),
                             inplace=True)  # Changed to loop for filling NaN

        # Feature scaling
        columns_to_scale = ['station_index', 'latitude', 'longitude',
                            'mekadem_nipuach_luz',
                            'passengers_continue_menupach', 'time_in_station',
                            'bus_capacity_at_arrival']
        data[columns_to_scale] = self.scaler.fit_transform(
            data[columns_to_scale])

        if is_training:
            X = data.drop(columns=['passengers_up'])
            y = data['passengers_up']
            return X, y, trip_id_unique_station
        else:
            return data, trip_id_unique_station

    def preprocess_data2(self, data, is_training=True):
        # Initialize trip_time to 0
        # Separate trip_id_unique_station to handle it independently

        first_arrival = data[data['station_index'] == 1].copy()
        # Find the last arrival times for each trip
        last_arrival = data.loc[data.groupby('trip_id_unique')['station_index'].idxmax()][
            ['trip_id_unique', 'arrival_time']]
        # Rename the last arrival time column for clarity
        last_arrival = last_arrival.rename(
            columns={'arrival_time': 'last_arrival_time'})
        # Merge the first arrival times with the last arrival times
        merged = pd.merge(first_arrival, last_arrival, on='trip_id_unique')
        # Calculate the trip time
        merged['trip_duration_in_minutes'] = (pd.to_datetime(
            merged['last_arrival_time']) - pd.to_datetime(
            merged['arrival_time'])).dt.total_seconds() / 60

        # Select the desired columns, keeping all original columns from the first arrival and adding trip_time
        data = merged.drop(columns=['last_arrival_time'])

        # Save and drop 'trip_id_unique'
        trip_id_unique = data['trip_id_unique'].copy()
        data.drop(columns=['trip_id_unique'], inplace=True)

        # Drop unnecessary columns
        drop_columns = ['line_id', 'part', 'station_id',
                        'trip_id_unique_station', 'station_name', 'alternative',
                        'passengers_continue_menupach']

        data.drop(columns=drop_columns, inplace=True)

        # Convert time columns to datetime
        time_columns = ['arrival_time', 'door_closing_time']
        data[time_columns] = data[time_columns].apply(pd.to_datetime,
                                                      format='%H:%M:%S',
                                                      errors='coerce')

        # Calculate 'time_in_station'
        data['time_in_station'] = (data['door_closing_time'] - data[
            'arrival_time']).dt.total_seconds()

        # Encode categorical variables
        categorical_columns = ['direction', 'cluster']
        label_encoders = {col: LabelEncoder() for col in categorical_columns}
        for col in categorical_columns:
            data[col] = label_encoders[col].fit_transform(data[col])

        # Convert 'arrival_is_estimated' to int
        data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)

        # Drop rows with null 'arrival_time'
        data.dropna(subset=['arrival_time'], inplace=True)

        # Replace '#' with NaN
        data.replace('#', np.nan, inplace=True)

        # Handle negative values in 'time_in_station'
        data.loc[data[
                     'time_in_station'] < 0, 'time_in_station'] = np.nan  # Changed to use .loc
        average_time_in_station = data['time_in_station'].mean(skipna=True)
        data['time_in_station'].fillna(average_time_in_station,
                                       inplace=True)  # Changed to use .fillna

        # Drop 'arrival_time' and 'door_closing_time' columns
        data.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

        # Drop 'passengers_continue'
        data.drop(columns=['passengers_continue'], inplace=True)

        # Convert all columns to numeric, forcing errors to NaN
        for col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')

        # Handle missing values by filling with the median value of the column
        for col in data.columns:
            data[col].fillna(data[col].median(), inplace=True)

        data.drop(columns=['station_index'], inplace=True)
        # Feature scaling
        columns_to_scale = ['latitude', 'longitude',
                            'mekadem_nipuach_luz',
                             'time_in_station' ]
        data[columns_to_scale] = self.scaler.fit_transform(
            data[columns_to_scale])

        if is_training:
            X = data.drop(columns=['trip_duration_in_minutes'])
            y = data['trip_duration_in_minutes']
            return X, y, trip_id_unique
        else:
            return data, trip_id_unique
