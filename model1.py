import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from dotenv import load_dotenv
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize

from config import config
from data import get_data

load_dotenv()

EXPERIMENT_NAME = "digits-classification"


def save_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Logistic Regression")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def train(model, x_train, y_train) -> None:
    model.fit(x_train, y_train)


def test(model, x_test, y_test) -> None:
    y_pred = model.predict(x_test)
    y_prob = model.predict_proba(x_test)
    class_names = [str(i) for i in range(10)]

    accuracy = accuracy_score(y_true=y_test, y_pred=y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    y_test_bin = label_binarize(y_test, classes=list(range(10)))
    auc_roc = roc_auc_score(y_test_bin, y_prob, multi_class="ovr", average="weighted")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_weighted", f1)
    mlflow.log_metric("auc_roc_weighted", auc_roc)

    cm = confusion_matrix(y_test, y_pred)
    with tempfile.TemporaryDirectory() as tmpdir:
        cm_path = os.path.join(tmpdir, "confusion_matrix.png")
        save_confusion_matrix(cm, class_names, cm_path)
        mlflow.log_artifact(cm_path)

    print(f"Logistic Regression — accuracy: {accuracy:.4f}, f1: {f1:.4f}, auc_roc: {auc_roc:.4f}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)

    logistic_regression_model = LogisticRegression(
        max_iter=config["logistic_regression"]["max_iter"],
    )

    data = get_data()

    with mlflow.start_run(run_name="logistic-regression"):
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("C", logistic_regression_model.C)
        mlflow.log_param("solver", logistic_regression_model.solver)
        mlflow.log_param("max_iter", logistic_regression_model.max_iter)

        train(logistic_regression_model, data["x_train"], data["y_train"])
        test(logistic_regression_model, data["x_test"], data["y_test"])
