import pandas as pd
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
import torch
from torch.utils.data import Dataset
from sklearn import metrics
import numpy as np
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns

class CFG:
    output_dir="./results_model_v2" # Directory for saving results
    eval_strategy="epoch"     # Evaluate at the end of each epoch
    learning_rate=5e-5              # Initial learning rate
    per_device_train_batch_size=16  # Batch size per GPU
    num_train_epochs=3              # Number of epochs
    weight_decay=0.01               # Regularization
    logging_steps=10                # Log every 10 steps

class DatasetReader:
    """Reads CSV dataset from disk into a pandas DataFrame."""
    def __init__(self, file_path, row_count):
        self.file_path = file_path
        self.row_count = row_count
        self.split = pd.DataFrame()
        self.load_data()
    
    def load_data(self):
        """Loads CSV file specified in self.file_path into self.split."""
        self.split = pd.read_csv(self.file_path).sample(n=self.row_count, random_state=42)

class TextDataset(Dataset):
    def __init__(self, df, text_col="text", label_col="label", max_length=256):
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.encodings = self.tokenizer(
            df[text_col].tolist(),
            padding="max_length",
            truncation=True,
            max_length=max_length
        )
        self.labels = df[label_col].tolist()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item
    
class Model:
    def __init__(self, CFG, train_dataset, test_dataset):
        self.cfg = CFG
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.model = DistilBertForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )   
                                
    def train(self):
        self.training_args = TrainingArguments(
            output_dir=self.cfg.output_dir,
            eval_strategy=self.cfg.eval_strategy,
            learning_rate=self.cfg.learning_rate,
            per_device_train_batch_size=self.cfg.per_device_train_batch_size,
            num_train_epochs=self.cfg.num_train_epochs,
            weight_decay=self.cfg.weight_decay,
            logging_steps=self.cfg.logging_steps
        )
        
        self.trainer = Trainer(
            model=self.model,                          # The DistilBERT model
            args=self.training_args,                   # Training arguments
            train_dataset=self.train_dataset,          # Training data
            eval_dataset=self.test_dataset,            # Validation data
            compute_metrics=self.compute_metrics,      # Eval function to compute acc and f1 score
        )

        self.trainer.train()

    @staticmethod
    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=1)
        acc = metrics.accuracy_score(labels, predictions)
        f1 = metrics.f1_score(labels, predictions)
        return {"accuracy": acc, "f1": f1}
  
    def evaluate(self):
        mlflow.start_run()

        results = self.trainer.predict(self.test_dataset)
        y_pred = np.argmax(results.predictions, axis=1)
        y_test = results.label_ids
        
        mlflow.set_tag("model_type", "transformer")

        mlflow.log_param("eval_strategy", self.cfg.eval_strategy)
        mlflow.log_param("learning_rate", self.cfg.learning_rate)
        mlflow.log_param("per_device_train_batch_size", self.cfg.per_device_train_batch_size)
        mlflow.log_param("num_train_epochs", self.cfg.num_train_epochs)
        mlflow.log_param("weight_decay", self.cfg.weight_decay)
        mlflow.log_param("logging_steps", self.cfg.logging_steps)

        conf_matrix = metrics.confusion_matrix(y_test, y_pred)
        acc_score = metrics.accuracy_score(y_test, y_pred)
        f1_score = metrics.f1_score(y_test, y_pred)
        clas_report = metrics.classification_report(y_test, y_pred, labels=[0, 1])

        mlflow.log_metric("acc_score", acc_score)
        mlflow.log_metric("f1_score", f1_score)

        fig, ax = plt.subplots()
        sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["negative", "positive"],
                    yticklabels=["negative", "positive"], ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        mlflow.log_figure(fig, "confusion_matrix.png")
        plt.close(fig)

        mlflow.log_text(clas_report, "classification_report.txt")

        mlflow.transformers.log_model(
            transformers_model={"model": self.model, "tokenizer": self.train_dataset.tokenizer},
            name="model",
            registered_model_name="imdb-sentiment-v2"
        )

        mlflow.end_run()


if __name__ == "__main__":
    train_data = DatasetReader('data/train.csv', 5000).split
    test_data = DatasetReader('data/test.csv', 1000).split

    train_dataset = TextDataset(train_data)
    test_dataset = TextDataset(test_data)

    # Configure MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("imdb-sentiment")

    model = Model(CFG, train_dataset, test_dataset)
    model.train()
    model.evaluate()
    