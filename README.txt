Files List and Description
Root Directory
README.txt - This file. Contains a list of all the files submitted and a brief description of each.
USERS.txt - Contains the team members’ logins and IDs. Format: login, ID.
project.pdf - A description of the project, including an explanation of the solution,
work process, and design decisions.
requirements.txt - Lists all the Python packages required to run the project code.


Dataset
train_bus_schedule.csv - The training data used for model fitting.
X_passengers_up.csv - Test data for the passengers up sub-task.
X_trip_duration.csv - Test data for the trip duration sub-task.
bus_column_description.md - A description of the dataset's features.

Predictions Directory
predictions - Directory containing the prediction .csv files for each sub-task.
X_passengers_up_predictions.csv - Predictions for the passengers up sub-task.
X_trip_duration_predictions.csv - Predictions for the trip duration sub-task.

Code Directory
code - Directory containing the main scripts and additional code files.
data_preprocessor.py - Contains functions for loading, preprocessing, and cleaning the data.
main_subtask1.py - Script for the trip duration sub-task. Includes the entire run flow:
loading data, preprocessing, model training, prediction, and saving predictions.
main_subtask2.py - Script for the passengers up sub-task. Includes the entire run flow:
loading data, preprocessing, model training, prediction, and saving predictions.
