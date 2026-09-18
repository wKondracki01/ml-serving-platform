"""Loads the v2 sentiment model (fine-tuned DistilBERT) from the MLflow Model
Registry and runs predictions on raw text.
"""

import mlflow

MAX_LENGTH = 256


class SentimentModel:
    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.pipeline = mlflow.transformers.load_model(model_uri)

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (sentiment_label, confidence) for a single piece of text."""
        result = self.pipeline(text, truncation=True, max_length=MAX_LENGTH)[0]
        label_idx = int(result["label"].split("_")[1])
        sentiment = "positive" if label_idx == 1 else "negative"
        confidence = float(result["score"])
        return sentiment, confidence
