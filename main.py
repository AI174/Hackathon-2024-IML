import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load the data with specified encoding
train_data = pd.read_csv('train_bus_schedule.csv', encoding='ISO-8859-8')

# Split the data into training and validation sets
train, valid = train_test_split(train_data, test_size=0.2, random_state=42)

# Combine train and validation data for consistent preprocessing
combined_data = pd.concat([train, valid], sort=False)

# Convert categorical features into numerical values
label_encoders = {}
categorical_features = ['part', 'trip_id_unique_station', 'trip_id_unique', 'alternative', 'cluster', 'station_name']

for feature in categorical_features:
    label_encoders[feature] = LabelEncoder()
    combined_data[feature] = label_encoders[feature].fit_transform(combined_data[feature])

# Split the combined data back into train and validation sets
train = combined_data.loc[train.index]
valid = combined_data.loc[valid.index]

# Fill missing values
imputer = SimpleImputer(strategy='mean')
train['passengers_up'] = imputer.fit_transform(train[['passengers_up']])
valid['passengers_up'] = imputer.transform(valid[['passengers_up']])

# Extract features and labels
features = ['trip_id', 'line_id', 'direction', 'station_index', 'station_id', 'latitude', 'longitude']
X_train = train[features]
y_train = train['passengers_up']
X_valid = valid[features]
y_valid = valid['passengers_up']

# Train in phases
percentages = [10, 20, 40, 60, 80, 100]
mse_scores = []

for percentage in percentages:
    if percentage < 100:
        partial_X_train, _, partial_y_train, _ = train_test_split(X_train, y_train, train_size=percentage / 100,
                                                                  random_state=42)
    else:
        partial_X_train = X_train
        partial_y_train = y_train

    model = RandomForestRegressor(random_state=42)
    model.fit(partial_X_train, partial_y_train)

    y_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_pred)
    mse_scores.append(mse)
    print(f'Training with {percentage}% of data - MSE: {mse}')

print(f'Mean Squared Error at different training sizes: {mse_scores}')

# Train the final model on the entire training set
final_model = RandomForestRegressor(random_state=42)
final_model.fit(X_train, y_train)

# Make predictions on the entire training set
train['passengers_up_pred'] = final_model.predict(X_train)

# Save the predictions to a CSV file
predictions = train[['trip_id_unique_station', 'passengers_up_pred']]
predictions.to_csv('passengers_up_predictions.csv', index=False)
