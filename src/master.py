import pandas as pd
import xgboost as xgb

from sklearn import metrics
from sklearn import preprocessing

def run(fold):
    #load the full training data with num_folds
    df = pd.read_csv("input/train_folds.csv")

    #list of numerical features/columns
    num_cols = [f"f{x}".format(x) for x in range(0,285)]            #columns - f0 to f284

    # map targets to 0s and 1s
    target_mapping = {
        "<=50k": 0,
        ">50k": 1
    }
    #df.loc[:, "target"] = df.target.map(target_mapping)            # applicable only when target column has values other than binaries

    # all columns are features except kfold & target column
    features = [f for f in df.columns if f not in ("kfold", "target")]

    # fill all NaN values with NONE
    # Note than I am converting all categorical columns to "strings"
    for col in features:
        # do not encode numeric columns
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna("NONE")

    # Now it is time to label encode the features

    for col in features:
        if col not in num_cols:
            # initialize LabelEncoder for each categorical feature column
            lbl = preprocessing.LabelEncoder()

            # fit LabelEncoder on all data
            lbl.fit(df[col])

            # transform all the data
            df.loc[:, col] = lbl.transform(df[col])

    # get training data using folds
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # get validation data using folds
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # get training data
    x_train = df_train[features].values

    # get validation data
    x_valid = df_valid[features].values

    # initialize xgboost model
    model = xgb.XGBClassifier(
        n_jobs = -1
    )

    # fit the model on training data
    model.fit(x_train, df_train.target.values)

    # predict on validation data
    # we need the probability values as we are calculating AUC
    # we will the probability of 1s
    valid_preds = model.predict_proba(x_valid)[:, 1]

    # get roc auc score
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)

    # print auc
    print(f"Fold = {fold}, AUC = {auc}")

if __name__ == "__main__":
    for fold_ in range(5):
        run(fold_)


