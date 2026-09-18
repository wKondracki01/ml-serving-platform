"""Sends IMDB test reviews to api-v1 and api-v2 at random, so their live
behavior (accuracy, confidence, latency) can be compared side by side.
"""

import argparse
import random

import pandas as pd
import requests

LABEL_TO_SENTIMENT = {0: "negative", 1: "positive"}


def run_ab_test(v1_url: str, v2_url: str, n: int, seed: int) -> pd.DataFrame:
    sample = pd.read_csv("training/data/test.csv").sample(n=n, random_state=seed)

    rows = []
    for _, row in sample.iterrows():
        model_version = random.choice(["v1", "v2"])
        base_url = v1_url if model_version == "v1" else v2_url

        response = requests.post(f"{base_url}/predict", json={"text": row["text"]})
        response.raise_for_status()
        result = response.json()

        rows.append(
            {
                "model_version": model_version,
                "true_sentiment": LABEL_TO_SENTIMENT[row["label"]],
                "predicted_sentiment": result["sentiment"],
                "confidence": result["confidence"],
                "latency_ms": result["latency_ms"],
                "correct": LABEL_TO_SENTIMENT[row["label"]] == result["sentiment"],
            }
        )

    return pd.DataFrame(rows)


def print_summary(results: pd.DataFrame) -> None:
    summary = results.groupby("model_version").agg(
        requests=("correct", "count"),
        accuracy=("correct", "mean"),
        avg_confidence=("confidence", "mean"),
        avg_latency_ms=("latency_ms", "mean"),
    )
    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=50, help="Number of requests to send")
    parser.add_argument("--v1-url", default="http://localhost:8000", help="Base URL for api-v1")
    parser.add_argument("--v2-url", default="http://localhost:8001", help="Base URL for api-v2")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling and routing")
    args = parser.parse_args()

    random.seed(args.seed)
    results = run_ab_test(args.v1_url, args.v2_url, args.n, args.seed)
    print_summary(results)
