import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

# Import test data.
print('Importing data...')
test_data = pd.read_csv('data/test_set_features.csv', index_col='respondent_id')

# Isolate categorical features and convert to category type
cat_features = test_data.select_dtypes(exclude=np.number).columns.tolist()

for feature in cat_features:
    test_data[feature] = test_data[feature].astype('category')

dtest_clf = xgb.DMatrix(test_data, enable_categorical=True)

# Import saved models
print('Importing pre-trained models...')
h1n1_file = 'models/h1n1_xgb_model.sav'
seasonal_file = 'models/seasonal_xgb_model.sav'

h1n1_model = pickle.load(open(h1n1_file, 'rb'))
seasonal_model = pickle.load(open(seasonal_file, 'rb'))

# Predict Probabilities
print('Generating predictions...')
h1n1_probability = h1n1_model.predict(dtest_clf)
seasonal_probability = seasonal_model.predict(dtest_clf)

# Create final submission csv
submission_df = pd.DataFrame()
submission_df['respondent_id'] = test_data.index
submission_df['h1n1_vaccine'] = h1n1_probability
submission_df['seasonal_vaccine'] = seasonal_probability
submission_df.to_csv('final_submission.csv', index=False)
