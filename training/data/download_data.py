"""Downloads and preprocesses the IMDB dataset for sentiment classification."""

from datasets import load_dataset
import pandas as pd
import re
import os

def preprocess_split(split):
    """Preprocess the text column values in datasets splits. Removes HTML regexes, double spaces
    and white spaces.
    """
    split = list(split)
    for sample in split:
        sample["text"] = re.sub(r"<.*?>", " ", sample["text"])
        sample["text"] = re.sub(r"\s+", " ", sample["text"])
        sample["text"] = sample["text"].strip()
    return split

def save_csv(split, data_dir, name):
    """Loads train split and test split into DF and saves as csv files."""
    split = pd.DataFrame(split)
    print(f"{name} total records: {len(split)}")
    print(f"{name} class percentage: {split['label'].value_counts()}")
    os.makedirs(data_dir, exist_ok=True)
    split.to_csv(f"{data_dir}/{name}.csv", index=False)

if __name__ == "__main__":

    DATA_DIR = "data/"

    dataset = load_dataset("imdb")
    train = dataset["train"]
    test = dataset["test"]
            
    train = preprocess_split(train)
    test = preprocess_split(test)

    save_csv(train, DATA_DIR, "train")
    save_csv(test, DATA_DIR, "test")
    print("Done! Files saved to training/data/")