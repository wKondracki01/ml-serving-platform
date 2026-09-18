"""Loads the v1 sentiment model (TF-IDF + Logistic Regression) from the
MLflow Model Registry and runs predictions on raw text.
"""

import mlflow


class SentimentModel:
    def __init__(self, model_uri: str):
        self.model_uri = model_uri
        self.pipeline = mlflow.sklearn.load_model(model_uri)

    def predict(self, text: str) -> tuple[str, float]:
        """Returns (sentiment_label, confidence) for a single piece of text."""
        proba = self.pipeline.predict_proba([text])[0]
        label_idx = int(proba.argmax())
        sentiment = "positive" if label_idx == 1 else "negative"
        confidence = float(proba[label_idx])
        return sentiment, confidence
