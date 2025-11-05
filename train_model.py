import os
from collections import Counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from ml.data import process_data
from ml.model import (
    compute_model_metrics,
    inference,
    load_model,
    performance_on_categorical_slice,
    save_model,
    train_model,
)

project_path = "."
data_path = os.path.join(project_path, "data", "census.csv")
data = pd.read_csv(data_path)

train, test = train_test_split(data, test_size=0.3, random_state=42, shuffle=True)

# Optional enhancement, use K-fold cross validation instead of a train-test split.
# my system is too weak to really support good K-fold CV right now, come back to this later

# DO NOT MODIFY
cat_features = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]

X_train, y_train, encoder, lb, std_scaler = process_data(
        train,
        categorical_features=cat_features,
        label="salary",
        training=True,
        encoder=OneHotEncoder,
        std_scaler=StandardScaler
    )

X_test, y_test, _, _, _ = process_data(
    test,
    categorical_features=cat_features,
    label="salary",
    training=False,
    encoder=encoder,
    lb=lb,
    std_scaler=std_scaler
)
# for hyperparameter tuning
parameters = {
    'C': [x/100 for x in range(1, 150, 9)],
    'penalty': ['l2'],
    'solver': ['liblinear', 'lbfgs', 'newton-cg', 'sag', 'saga'],
    'max_iter': [500, 750, 1000]
}

# post-tuning, known best parameters
# parameters = {
#     'C': [1.27],
#     'max_iter': [500],
#     'solver': ['lbfgs'],
#     'penalty': ['l2'],
#     'tol': [0.0001]
# }

model = train_model(
    X_train=X_train,
    y_train=y_train,
    parameters=parameters
)

# save the model and the encoder
model_path = os.path.join(project_path, "model", "model.pkl")
save_model(model, model_path)
encoder_path = os.path.join(project_path, "model", "encoder.pkl")
save_model(encoder, encoder_path)
std_scaler_path = os.path.join(project_path, "model", "stdscaler.pkl")
save_model(std_scaler, std_scaler_path)

# load the model
model = load_model(
    model_path
)

preds = inference(model=model, X=X_test) # type: ignore

# Calculate and print the metrics
p, r, fb = compute_model_metrics(y_test, preds)
print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}")

# compute the performance on model slices using the performance_on_categorical_slice function
# iterate through the categorical features
# prep the output csv file
metrics = {
    "metric": [],
    "value": [],
    "count": [],
    "precision": [],
    "recall": [],
    "f1": []
}

metrics["metric"].append("full_data")
metrics["value"].append("all")
metrics["count"].append(y_test.shape[0])
metrics["precision"].append(p)
metrics["recall"].append(r)
metrics["f1"].append(fb)

for col in cat_features:
    # iterate through the unique values in one categorical feature
    for slicevalue in sorted(test[col].unique()):
        count = test[test[col] == slicevalue].shape[0]
        p, r, fb = performance_on_categorical_slice(
            data=data,
            column_name=col,
            slice_value=slicevalue,
            categorical_features=cat_features,
            label='salary',
            encoder=encoder,
            lb=lb,
            std_scaler=std_scaler,
            model=model
        )
        with open("data/slice_output.txt", "a") as f:
            print(f"{col}: {slicevalue}, Count: {count:,}", file=f)
            print(f"Precision: {p:.4f} | Recall: {r:.4f} | F1: {fb:.4f}", file=f)

        metrics["metric"].append(col)
        metrics["value"].append(slicevalue)
        metrics["count"].append(count)
        metrics["precision"].append(p)
        metrics["recall"].append(r)
        metrics["f1"].append(fb)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv("data/slice_metrics.csv", index=False)        

# compute baseline accuracy vs actual accuracy
# find most common label in training set
majority_class = Counter(y_train).most_common(1)[0][0]

# create baseline preds for test set
y_pred_baseline = np.full_like(y_test, fill_value=majority_class)

# baseline accuracy
baseline_accuracy = accuracy_score(y_test, y_pred_baseline)
actual_accuracy = accuracy_score(y_test, preds)

print(f"Baseline Accuracy: {baseline_accuracy}")
print(f"Experimental Accuracy: {actual_accuracy}")