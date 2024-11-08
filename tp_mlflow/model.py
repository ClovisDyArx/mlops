import mlflow
from mlflow.models import infer_signature
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import utils_mlflow


def load_data():
    X, y = datasets.load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def define_params():
    params = {
        "solver": "lbfgs",
        "max_iter": 1000,
        "multi_class": "auto",
        "random_state": 8888,
    }

    return params


def model_training(params, X_train, X_test, y_train, y_test):
    logreg = LogisticRegression(**params)
    logreg.fit(X_train, y_train)

    y_pred = logreg.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    return logreg, accuracy


base_uri = utils_mlflow.get_base_uri()
mlflow.set_tracking_uri(uri=base_uri)

# Create a new MLflow Experiment
mlflow.set_experiment("MLflow Quickstart")

# Start an MLflow run
with mlflow.start_run():
    params = define_params()
    X_train, X_test, y_train, y_test = load_data()
    logreg, accuracy = model_training(params, X_train, X_test, y_train, y_test)

    # Log the hyperparameters
    mlflow.log_params(params)

    # Log the loss metric
    mlflow.log_metric("accuracy", accuracy)

    # Set a tag that we can use to remind ourselves what this run was for
    mlflow.set_tag("Training Info", "Basic logreg model for iris data")

    # Infer the model signature
    signature = infer_signature(X_train, logreg.predict(X_train))

    # Log the model
    model_info = mlflow.sklearn.log_model(
        sk_model=logreg,
        artifact_path="iris_model",
        signature=signature,
        input_example=X_train,
        registered_model_name="tp_mlflow_mlops",
    )

# Load the model back for predictions as a generic Python Function model
loaded_model = mlflow.pyfunc.load_model(model_info.model_uri)

predictions = loaded_model.predict(X_test)

iris_feature_names = datasets.load_iris().feature_names

result = pd.DataFrame(X_test, columns=iris_feature_names)
result["actual_class"] = y_test
result["predicted_class"] = predictions

print(result[:4])
