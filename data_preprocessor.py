import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer

class DataPreprocessor:
    def __init__(self):
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.imputer = SimpleImputer(strategy='mean')

    def preprocess_data(self, data, is_training=True):
        drop_columns = ['line_id', 'part', 'trip_id_unique','station_id','station_name','station_index']
        data.drop(columns=drop_columns, inplace=True)

        # Encode categorical features ???????????????????????????????????????
        categorical_features = ['cluster']
        for feature in categorical_features:
            if feature not in self.label_encoders:
                self.label_encoders[feature] = LabelEncoder()
            if is_training:
                data[feature] = self.label_encoders[feature].fit_transform(data[feature])
            else:
                data[feature] = self.label_encoders[feature].transform(data[feature])

        data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
        data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce')
        data['up_passengers_time'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds()
        data.drop(columns=['door_closing_time'], inplace=True)

        # Convert boolean to int
        data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)
        data['alternative'] = data['alternative'].apply(lambda x: int(x) if isinstance(x, str) and x.isdigit() else
                                (x if isinstance(x, int) else 0))


        # Create new feature
        data['bus_capacity_at_arrival'] = data['passengers_continue'] + data['passengers_up']
        data.drop(columns=['passengers_continue'], inplace=True)

        # Fill missing values
        data = pd.DataFrame(self.imputer.fit_transform(data), columns=data.columns)

        # Feature scaling
        numerical_features = ['direction', 'alternative', 'station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                              'passengers_continue_menupach', 'time_in_station', 'bus_capacity_at_arrival']
        data[numerical_features] = self.scaler.fit_transform(data[numerical_features])

        if is_training:
            X = data.drop(columns=['passengers_up'])
            y = data['passengers_up']
            return X, y
        else:
            return data

    def preprocess_test(self, data):
        # Same preprocessing as train but without dropping target columns
        columns_to_drop = ['line_id', 'part', 'trip_id_unique', 'station_id', 'station_name']
        data.drop(columns=columns_to_drop, inplace=True)

        # Encode categorical features
        categorical_features = ['cluster']
        for feature in categorical_features:
            data[feature] = self.label_encoders[feature].transform(data[feature])

        # Process time columns
        data['arrival_time'] = pd.to_datetime(data['arrival_time'], format='%H:%M:%S', errors='coerce')
        data['door_closing_time'] = pd.to_datetime(data['door_closing_time'], format='%H:%M:%S', errors='coerce')
        data['time_in_station'] = (data['door_closing_time'] - data['arrival_time']).dt.total_seconds()
        data.drop(columns=['arrival_time', 'door_closing_time'], inplace=True)

        # Convert boolean to int
        data['arrival_is_estimated'] = data['arrival_is_estimated'].astype(int)

        # Create new feature
        data['bus_capacity_at_arrival'] = data['passengers_continue'] + data['passengers_up']
        data.drop(columns=['passengers_continue'], inplace=True)

        # Fill missing values
        data = pd.DataFrame(self.imputer.transform(data), columns=data.columns)

        # Feature scaling
        numerical_features = ['direction', 'alternative', 'station_index', 'latitude', 'longitude', 'mekadem_nipuach_luz',
                              'passengers_continue_menupach', 'time_in_station', 'bus_capacity_at_arrival']
        data[numerical_features] = self.scaler.transform(data[numerical_features])

        return data
