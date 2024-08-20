from argparse import ArgumentParser
import pandas as pd
from hackathon_code.data_preprocessor import Preprocessor
import numpy as np
from sklearn.linear_model import LinearRegression
import logging
from matplotlib import pyplot as plt
import seaborn as sns


### running commands:
# --training_set train_bus_schedule.csv --test_set X_passengers_up.csv --out passengers_up_predictions.csv


class LinearRegressModel:
    def __init__(self):
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X):
        predictions = self.model.predict(X)
        # Ensure no negative predictions
        predictions[predictions < 0] = 0
        return predictions

###################  PLOTS ##########################################

def plot_correlation_matrix(data_combined, filename='correlation_matrix.png'):
    # Calculate correlation matrix
    correlation_matrix = data_combined.corr()

    # Plot the correlation matrix
    plt.figure(figsize=(12, 12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix')
    plt.savefig(filename)
    plt.close()


def plot_feature_relationships(combined_data, output_dir='plots'):
    import os
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for feature in combined_data.columns:
        if feature != 'passengers_up':
            plt.figure(figsize=(10, 6))
            correlation = combined_data[feature].corr(
                combined_data['passengers_up'])
            plt.scatter(combined_data[feature], combined_data['passengers_up'],
                        alpha=0.5)
            plt.title(
                f'Relationship between {feature} and passengers_up\nCorrelation: {correlation:.2f}')
            plt.xlabel(feature)
            plt.ylabel('passengers_up')
            plt.grid(True)
            plt.savefig(f'{output_dir}/relationship_{feature}.png')
            plt.close()


def plot_passenger_trends(data, filename='passenger_trends.png'):
    # Extract the hour from the 'time_in_station' column
    data['hour'] = pd.to_datetime(data['arrival_time']).dt.hour

    # Group by hour and calculate the average number of passengers boarding
    hourly_data = data.groupby('hour')['passengers_up'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(hourly_data['hour'], hourly_data['passengers_up'], marker='o')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Number of Passengers Boarding')
    plt.title('Average Number of Passengers Boarding by Hour')
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.savefig(filename)
    plt.close()


def plot_passenger_trends_by_line(data,
                                  filename='passenger_trends_by_line.png'):
    # Extract the hour from the 'arrival_time' column
    data['hour'] = pd.to_datetime(data['arrival_time']).dt.hour

    # Group by hour and line_number, and calculate the average number of passengers boarding
    hourly_line_data = data.groupby(['hour', 'line_id'])[
        'passengers_up'].mean().reset_index()

    # Plotting
    plt.figure(figsize=(14, 8))
    sns.lineplot(data=hourly_line_data, x='hour', y='passengers_up',
                 hue='line_id', marker='o')

    plt.xlabel('Hour of the Day')
    plt.ylabel('Average Number of Passengers Boarding')
    plt.title('Average Number of Passengers Boarding by Hour and Line Number')
    plt.grid(True)
    plt.xticks(range(0, 24))
    plt.legend(title='Line Number', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.savefig(filename)
    plt.close()


def plot_bus_usage_region(data, filename='bus_usage_region.png'):
    region_data = data.groupby('cluster')['passengers_up'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(data=region_data, x='cluster', y='passengers_up',
                palette='viridis')
    plt.xlabel('Region')
    plt.ylabel('Total Passengers Boarding')
    plt.title('Bus Usage by Region')
    plt.xticks(rotation=45)
    plt.savefig(filename)
    plt.close()

def first_question_draw_plots(train):
    plot_correlation_matrix(train)
    plot_feature_relationships(train)
    plot_passenger_trends(train)
    plot_passenger_trends_by_line(train)
    plot_bus_usage_region(train)


def save_predictions(predictions, output_path, col, cols_names):
    # Ensure consistent lengths before saving
    if len(predictions) != len(col):
        min_length = min(len(predictions), len(col))
        predictions = predictions[:min_length]
        col = col[:min_length]

    # Round predictions to the nearest whole number
    rounded_predictions = np.round(predictions).astype(int)
    output = pd.DataFrame({cols_names[0]: col,
                           cols_names[1]: rounded_predictions})
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
    X_train, y_train, trip_id_unique_station_train = \
        preprocessor.preprocess_data(train_data, is_training=True)

    # 3. train a model
    logging.info("training...")
    model = LinearRegressModel()
    model = model.train(X_train, y_train)  # Corrected: Do not reassign `model`

    # 4. load the test set (args.test_set)
    test_data = pd.read_csv(args.test_set, encoding='ISO-8859-1')

    # 5. preprocess the test set
    logging.info("preprocessing test...")
    X_test, trip_id_unique_station_test = preprocessor.preprocess_data(
        test_data, is_training=False)

    # 6. predict the test set using the trained model
    logging.info("predicting...")
    predictions = model.predict(X_test)
    predictions[predictions < 0] = 0

    # 7. save the predictions to args.out
    logging.info("predictions saved to {}".format(args.out))
    cols_names = ['trip_id_unique_station', 'passengers_up']
    save_predictions(predictions, args.out, trip_id_unique_station_test,
                     cols_names)
    # 8. plotting :
    first_question_draw_plots(train_data)