"""Training script for IMDB sentiment classification model v1 (TF-IDF + Logistic Regression).

Loads preprocessed IMDB data, vectorizes text using TF-IDF,
trains a Logistic Regression model, and logs results to MLflow.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn import metrics
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

class CFG:
    """Hyperparameters for Logistic Regression model."""
    penalty = 'l2'
    C = 10.0
    solver = 'liblinear'
    max_iter = 1000

class DatasetReader:
    """Reads CSV dataset from disk into a pandas DataFrame."""
    def __init__(self, file_path):
        self.file_path = file_path
        self.split = pd.DataFrame()
        self.load_data()
    
    def load_data(self):
        """Loads CSV file specified in self.file_path into self.split."""
        self.split = pd.read_csv(self.file_path)

class Model:
    """Logistic Regression model with MLflow tracking integration."""
    def __init__(self, X_train, y_train, X_test, y_test, CFG):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.cfg = CFG
        self.model  = Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(
                penalty = self.cfg.penalty,
                C = self.cfg.C,
                solver = self.cfg.solver,
                max_iter = self.cfg.max_iter
            )),
        ])
        self.acc_score = None
        self.f1_score = None
        self.conf_matrix = None
        self.clas_report = None
    
    def train(self):
        """Fits the model on training data."""
        self.model = self.model.fit(self.X_train, self.y_train)
    
    def predict(self):
        """Returns predictions on test data."""
        return self.model.predict(self.X_test)
    
    def evaluate(self):
        """Evaluates model performance and logs everything to MLflow.

        Logs: hyperparameters, accuracy, F1 score, confusion matrix (as image),
        classification report (as text), and registers the model in MLflow Registry.
        """
        mlflow.start_run()

        mlflow.log_param("penalty", self.cfg.penalty)
        mlflow.log_param("C", self.cfg.C)
        mlflow.log_param("solver", self.cfg.solver)
        mlflow.log_param("max_iter", self.cfg.max_iter)

        y_pred = self.predict()
        self.conf_matrix = metrics.confusion_matrix(self.y_test, y_pred)
        self.acc_score = metrics.accuracy_score(self.y_test, y_pred)
        self.f1_score = metrics.f1_score(self.y_test, y_pred)
        self.clas_report = metrics.classification_report(self.y_test, y_pred, labels=[0, 1])

        mlflow.log_metric("acc_score", self.acc_score)
        mlflow.log_metric("f1_score", self.f1_score)

        fig, ax = plt.subplots()
        sns.heatmap(self.conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["negative", "positive"],
                    yticklabels=["negative", "positive"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.log_text(self.clas_report, "classification_report.txt")

        mlflow.sklearn.log_model(self.model, name="model", registered_model_name="imdb-sentiment-v1")

        mlflow.end_run()
        
if __name__ == "__main__":
    # Load preprocessed data
    train_data = DatasetReader("data/train.csv").split
    test_data = DatasetReader("data/test.csv").split

    # Extract labels
    y_train = train_data["label"]
    y_test = test_data["label"]

    # Configure MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("imdb-sentiment")

    # Train and evaluate
    cfg = CFG()
    model = Model(train_data["text"], y_train, test_data["text"], y_test, cfg)
    model.train()
    model.evaluate()    