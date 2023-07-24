from sklearn.metrics import roc_auc_score


def evaluate(model, vaccine, X_test, y_test):
    y_pred = model.predict(X_test)
    roc_auc = roc_auc_score(y_test[vaccine], y_pred)
    print(f'The ROC-AUC score for this model is {roc_auc}')
    return (roc_auc)
    