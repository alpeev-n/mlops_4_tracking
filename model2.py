import os
import tempfile

import matplotlib.pyplot as plt
import mlflow
import seaborn as sns
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize
from sklearn.tree import DecisionTreeClassifier

from config import config
from data import get_data

load_dotenv()

EXPERIMENT_NAME = "digits-classification"


def save_confusion_matrix(cm, class_names, path):
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Decision Tree")
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

    print(f"Decision Tree — accuracy: {accuracy:.4f}, f1: {f1:.4f}, auc_roc: {auc_roc:.4f}")


if __name__ == "__main__":
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(EXPERIMENT_NAME)

    decision_tree_model = DecisionTreeClassifier(
        random_state=config["random_state"],
        max_depth=config["decision_tree"]["max_depth"],
        criterion=config["decision_tree"]["criterion"],
        min_samples_leaf=config["decision_tree"]["min_samples_leaf"],
    )

    data = get_data()

    with mlflow.start_run(run_name="decision-tree"):
        mlflow.log_param("model_type", "DecisionTree")
        mlflow.log_param("max_depth", decision_tree_model.max_depth)
        mlflow.log_param("min_samples_leaf", decision_tree_model.min_samples_leaf)
        mlflow.log_param("criterion", decision_tree_model.criterion)
        mlflow.log_param("random_state", decision_tree_model.random_state)

        train(decision_tree_model, data["x_train"], data["y_train"])
        mlflow.log_param("n_leaves", decision_tree_model.get_n_leaves())
        mlflow.log_param("depth", decision_tree_model.get_depth())
        test(decision_tree_model, data["x_test"], data["y_test"])
