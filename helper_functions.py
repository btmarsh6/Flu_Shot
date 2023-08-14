from sklearn.metrics import roc_auc_score

def clean_data(df):
    # Drop columns with large proportion of missing values.
    df = df.drop(columns=['employment_industry', 'employment_occupation', 'health_insurance'])

    # Drop rows with missing values
    df = df.dropna()

    return df

def evaluate(model, vaccine, X_test, y_test):
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test[vaccine], y_pred)
    print(f'The ROC-AUC score for the model predicting {vaccine} uptake is {roc_auc}')
    return roc_auc

def xgb_evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'The ROC-AUC score for the {model} is {roc_auc}')
    return roc_auc