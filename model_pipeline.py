import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from helper_functions import evaluate, clean_data
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
import pickle

# Import training data.
print('Importing data...')
training_data = pd.read_csv('data/training_set_features.csv', index_col='respondent_id')
training_labels = pd.read_csv('data/training_set_labels.csv', index_col='respondent_id')

print('Cleaning data...')
training_data = clean_data(training_data)

# Drop corresponding rows from training labels.
mask = training_labels.index.isin(training_data.index)
clean_labels = training_labels[mask]

# Isolate categorical and numerical features.
cat_features = training_data.select_dtypes(exclude=np.number).columns.tolist()
num_features = training_data.select_dtypes(np.number).columns.tolist()

# Split training data into training and testing.
X_train, X_test, y_train, y_test = train_test_split(training_data, clean_labels, test_size=.2, random_state=58)


# Build Pipeline
print('Building pipeline...')


numeric_transform = Pipeline([
    ('scaler', MinMaxScaler())
])

categorical_transform = Pipeline([
    ('encode', OneHotEncoder(drop='first'))
])

preprocessing = ColumnTransformer([
    ('numeric', numeric_transform, num_features),
    ('categorical', categorical_transform, cat_features)
])

h1n1_gnb_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', GaussianNB())
])
seasonal_svc_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', SVC(C=1, gamma='auto', probability=True))
])

# Train model
print('Training models...')
h1n1_gnb_pipeline.fit(X_train, y_train['h1n1_vaccine'])
seasonal_svc_pipeline.fit(X_train, y_train['seasonal_vaccine'])

# Evaluate model
print('Evaluating models...')
score_h1n1 = evaluate(h1n1_gnb_pipeline, 'h1n1_vaccine', X_test, y_test)
score_seasonal = evaluate(seasonal_svc_pipeline, 'seasonal_vaccine', X_test, y_test)

# Save models
save_h1n1_model = input('Would you like to save the H1N1 model? (y/n)')
if save_h1n1_model == 'y':
    h1n1_filename = input('Enter filename for model: ')
    pickle.dump(h1n1_gnb_pipeline, open(h1n1_filename, 'wb'))
    print('File saved!')

save_seasonal_model = input('Would you like to save the seasonal model? (y/n)')
if save_seasonal_model == 'y':
    seasonal_filename = input('Enter filename for model: ')
    pickle.dump(seasonal_svc_pipeline, open(seasonal_filename, 'wb'))
    print('File saved!')
    