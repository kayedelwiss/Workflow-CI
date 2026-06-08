import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train = pd.read_csv('dataset_preprocessing/X_train.csv')
X_test = pd.read_csv('dataset_preprocessing/X_test.csv')

y_train = pd.read_csv('dataset_preprocessing/y_train.csv')
y_test = pd.read_csv('dataset_preprocessing/y_test.csv')

mlflow.set_experiment("Stroke Prediction")

mlflow.sklearn.autolog()

mlflow.set_experiment("Stroke Prediction")
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train.values.ravel())

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print("Accuracy:", accuracy)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "model")