import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from helper_functions import evaluate
from sklearn.svm import SVC


# Import training data.
print('Importing data...')
training_data = pd.read_csv('data/training_set_features.csv', index_col='respondent_id')
training_labels = pd.read_csv('data/training_set_labels.csv', index_col='respondent_id')

print('Cleaning data...')
# Drop columns with large proportion of missing values.
training_data = training_data.drop(columns=['employment_industry', 'employment_occupation', 'health_insurance'])

# Drop rows with missing values
training_data = training_data.dropna()

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

h1n1_svc_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', SVC())
])
seasonal_svc_pipeline = Pipeline([
    ('preprocessing', preprocessing),
    ('model', SVC())
])

param_grid = {'model__C': [.75, 1, 1.25],
              'model__gamma': ['scale', 'auto', 1]
              }

# Hyperparameter tuning
h1n1_grid = GridSearchCV(h1n1_svc_pipeline, param_grid=param_grid, verbose=10, scoring='roc_auc')
seasonal_grid = GridSearchCV(seasonal_svc_pipeline, param_grid=param_grid, verbose=10, scoring='roc_auc')


# Train model
print('Training models...')
h1n1_grid.fit(X_train, y_train['h1n1_vaccine'])
seasonal_grid.fit(X_train, y_train['seasonal_vaccine'])

# Evaluate model
print('Evaluating models...')
score_h1n1 = evaluate(h1n1_grid, 'h1n1_vaccine', X_test, y_test)
score_seasonal = evaluate(seasonal_grid, 'seasonal_vaccine', X_test, y_test)

h1n1_best_params = h1n1_grid.get_params()
seasonal_best_params = seasonal_grid.get_params()
print(f'Best parameters for h1n1 model: {h1n1_best_params}\nBest parameters for seasonal model: {seasonal_best_params}')