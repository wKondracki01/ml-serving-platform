import pandas as pd
import mlflow
from sklearn import metrics

if __name__ == "__main__":
    # Configure MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("imdb-sentiment")

    # Load the same test sample that v2 was evaluated on, so both models
    # are scored on exactly the same rows.
    test_data = pd.read_csv("data/test.csv").sample(n=1000, random_state=42)
    texts = test_data["text"]
    y_true = test_data["label"]

    # Load v1 and v2 model from Registry
    model_v1 = mlflow.sklearn.load_model("models:/imdb-sentiment-v1/1")
    model_v2 = mlflow.transformers.load_model("models:/imdb-sentiment-v2/1")

    # Predictions for two of the models
    y_pred_v1 = model_v1.predict(texts)
    raw = model_v2(texts.tolist())
    y_pred_v2 = [int(r["label"].split("_")[1]) for r in raw]

    # Metrics
    mlflow.start_run(run_name="v1-vs-v2-comparison")

    acc_score_v1 = metrics.accuracy_score(y_true, y_pred_v1)
    acc_score_v2 = metrics.accuracy_score(y_true, y_pred_v2)

    f1_score_v1 = metrics.f1_score(y_true, y_pred_v1)
    f1_score_v2 = metrics.f1_score(y_true, y_pred_v2)

    clas_report_v1 = metrics.classification_report(y_true, y_pred_v1, labels=[0, 1])
    clas_report_v2 = metrics.classification_report(y_true, y_pred_v2, labels=[0, 1])

    mlflow.log_metric("acc_score_v1", acc_score_v1)
    mlflow.log_metric("acc_score_v2", acc_score_v2)

    mlflow.log_metric("f1_score_v1", f1_score_v1)
    mlflow.log_metric("f1_score_v2", f1_score_v2)

    mlflow.log_text(clas_report_v1, "classification_report_v1.txt")
    mlflow.log_text(clas_report_v2, "classification_report_v2.txt")

    mlflow.end_run()
