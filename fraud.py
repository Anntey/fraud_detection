
#############
# Libraries #
#############

import shap
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, precision_recall_curve, roc_curve, confusion_matrix, average_precision_score
from inspect import signature

################
# Loading data #
################

train_transaction = pd.read_csv("./input/train_transaction.csv", index_col = "TransactionID")
test_transaction = pd.read_csv("./input/test_transaction.csv", index_col = "TransactionID")

train_identity = pd.read_csv("./input/train_identity.csv", index_col = "TransactionID")
test_identity = pd.read_csv("./input/test_identity.csv", index_col = "TransactionID")

subm_df = pd.read_csv("./input/sample_submission.csv", index_col = "TransactionID")

train = train_transaction.merge(train_identity, how = "left", left_index = True, right_index = True) # left join identities to actions
test = test_transaction.merge(test_identity, how = "left", left_index = True, right_index = True)

del train_identity, test_identity # clear some memory

y_train = train["isFraud"].copy()
x_train = train.drop("isFraud", axis = 1)

x_test = test.copy()

x_train = x_train.fillna(-999)
x_test = x_test.fillna(-999)

for col in x_train.columns:
    if x_train[col].dtype == "object" or x_test[col].dtype == "object":  # label Encoding
        label = LabelEncoder()
        label.fit(list(x_train[col].values) + list(x_test[col].values))
        x_train[col] = label.transform(list(x_train[col].values))
        x_test[col] = label.transform(list(x_test[col].values))  

#x_train, x_val, y_train, y_val = train_test_split(
#        x_train,
#        y_train,
#        test_size = 0.2,
#        stratify = train["isFraud"], # preserve proportions
#        random_state = 2019
#)

del train, test

#################
# Visualization #
#################

sns.barplot([0, 1], train_transaction["isFraud"].value_counts()) # unbalanced data
plt.title("Fraud count")

del train_transaction, test_transaction 

#########################
# Reducing memory usage #
#########################
        
def reduce_mem_usage(df):
    """Iterate through all the columns of a dataframe and modify the data type to reduce memory usage."""
    
    start_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage of dataframe is {:.2f} MB".format(start_mem))
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype("category")

    end_mem = df.memory_usage().sum() / 1024**2
    print("Memory usage after optimization is: {:.2f} MB".format(end_mem))
    print("Decreased by {:.1f}%".format(100 * (start_mem - end_mem) / start_mem))
    
    return df

x_train = reduce_mem_usage(x_train)
x_test = reduce_mem_usage(x_test)

#################
# Fitting model #
#################

train_xgb = xgb.DMatrix(x_train, y_train)
#val_xgb = xgb.DMatrix(x_val, y_val)
test_xgb = xgb.DMatrix(x_test)
         
params_xgb = {
        "gamma": 0.1,
        "learning_rate": 0.05,
        "max_depth": 9,
        "n_estimators": 500,
        "objective": "binary:logistic",
        "reg_lambda": 1,
        "subsample": 0.9,
        "colsample_bytree": 0.9,
        "missing": -999,
        "random_state": 2019
}

model = xgb.train(
        params_xgb,
        train_xgb,
#       evals = [(train_xgb, "train"), (val_xgb, "valid")],
#       early_stopping_rounds = 500,
        num_boost_round = 400
)

##############
# Evaluation #
##############

# SHAP values
shap_explainer = shap.TreeExplainer(model)
shap_values = shap_explainer.shap_values(x_train[:50000]) # data set too large as a whole
shap.summary_plot(shap_values, x_train[:50000]) # feature importance summary

# validation set prediction
y_pred = model.predict(val_xgb)
y_true = y_val

# Confusion matrix
cm = confusion_matrix(y_true, y_pred > 0.5)

plt.figure(figsize = (5, 4))
sns.heatmap(cm, cmap = "Blues", annot = True, fmt = "d")
plt.title("Confusion matrix")
plt.ylabel("True label")
plt.xlabel("Predicted label")

# ROC curve
f_pos_rate, t_pos_rate, thresholds = roc_curve(y_true, y_pred)
auc = roc_auc_score(y_true, y_pred)

plt.figure()
plt.plot(f_pos_rate, t_pos_rate, color = "darkorange", lw = 2, label = "ROC curve (AUC = %0.3f)" % auc)
plt.plot([0, 1], [0, 1], color = "navy", lw = 2, linestyle = "--")
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.title("ROC")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc = "lower right")
plt.show()

# Precision-Recall curve
precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
average_precision = average_precision_score(y_true, y_pred)

step_kwargs = ({"step": "post"} if "step" in signature(plt.fill_between).parameters else {})
plt.step(recall, precision, color = "b", alpha = 0.3, where = "post")
plt.fill_between(recall, precision, alpha = 0.3, color = "b", **step_kwargs)
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.05])
plt.title("Precision-Recall curve (avg precision = {0:0.3f})".format(average_precision))
plt.xlabel("Recall")
plt.ylabel("Precision")


##############
# Prediction #
##############

preds = model.predict(test_xgb)

subm_df["isFraud"] = preds

subm_df.to_csv("submission.csv")
