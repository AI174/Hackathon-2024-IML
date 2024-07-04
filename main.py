import pandas as pd
from sklearn.model_selection import train_test_split
from model import Model
from data_preprocessor import DataPreprocessor

# Load the data with specified encoding
train_data = pd.read_csv('train_bus_schedule.csv', encoding='ISO-8859-8')
test_data = pd.read_csv('X_passengers_up.csv', encoding='ISO-8859-8')

# Split the data into training and validation sets
train, valid = train_test_split(train_data, test_size=0.2, random_state=42)

# Initialize the preprocessor
preprocessor = DataPreprocessor()

# Preprocess the training and validation data
X_train, y_train, train_trip_ids = preprocessor.preprocess_data(train, is_training=True)
X_valid, y_valid, valid_trip_ids = preprocessor.preprocess_data(valid, is_training=True)

# Train the model in phases
model = Model()
percentages = [10, 20, 40, 60, 80, 100]
mse_scores = []

for percentage in percentages:
    if percentage < 100:
        partial_X_train, _, partial_y_train, _ = train_test_split(X_train, y_train, train_size=percentage / 100, random_state=42)
    else:
        partial_X_train = X_train
        partial_y_train = y_train

    model.fit(partial_X_train, partial_y_train)
    mse = model.loss(X_valid, y_valid)
    mse_scores.append(mse)
    print(f'Training with {percentage}% of data - MSE: {mse}')

print(f'Mean Squared Error at different training sizes: {mse_scores}')

# Train the final model on the entire training set
model.fit(X_train, y_train)

# Preprocess the test data
X_test, test_trip_ids = preprocessor.preprocess_test(test_data)

# Make predictions on the test set
test_data['passengers_up'] = model.predict(X_test)

# Reattach trip_id_unique_station to the test_data DataFrame
test_data['trip_id_unique_station'] = test_trip_ids

# Save the predictions to a CSV file
predictions = test_data[['trip_id_unique_station', 'passengers_up']]
predictions.to_csv('passengers_up_predictions.csv', index=False)
