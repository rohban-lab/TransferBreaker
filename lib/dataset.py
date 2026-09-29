from datasets import Dataset
import torch
from datasets import Audio
import pandas as pd
from .normalizer import Normalizer
from torch.utils.data import DataLoader
import os


# -----------------------------------------------------------------------------------------------
def load_dataset(dir_path: str, lang, split, frac=1.0, num_samples=None, seed=None) -> Dataset:

    if split not in ["train", "test", "dev"]:
        raise ValueError(f"invalid split: {split}")

    if not 0.0 < frac <= 1.0:
        raise ValueError(f"frac must be in (0, 1], got {frac}")
    if num_samples is not None and num_samples < 0:
        raise ValueError(f"num_samples must be >= 0, got {num_samples}")

    transcript_path = os.path.join(dir_path, split, "transcript.csv")
    df = pd.read_csv(transcript_path)
    df["file_path"] = df["file_name"].apply(
        lambda name: os.path.join(dir_path, split, "audio", name)
    )

    # num_samples overrides frac if specified. A seed of 0 is valid.
    if num_samples is not None:
        num_samples = min(int(num_samples), len(df))  # cap at dataset size
        if seed is not None:
            df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)
        else:
            df = df.iloc[:num_samples].copy()
    elif seed is not None:
        df = df.sample(frac=frac, random_state=seed).reset_index(drop=True)
    else:
        df = df.iloc[:int(frac * len(df))].copy()

    dataset = Dataset.from_pandas(df)
    dataset = dataset.cast_column("file_path", Audio(sampling_rate=16000))
    dataset = dataset.rename_column("file_path", "audio")

    normalizer = Normalizer(lang=lang)

    def prepare_func(row):
        audio = row["audio"]["array"].astype("float32")
        transcript = normalizer([row["transcript"]])[0]
        return {"audio": audio, "transcript": transcript}

    dataset = dataset.map(prepare_func)

    return dataset


# -----------------------------------------------------------------------------------------------
def create_dataloader(dataset: Dataset, batch_size: int) -> DataLoader:

    def collate_fn(batch):
        audio = [torch.tensor(row["audio"], dtype=torch.float32).unsqueeze(0) for row in batch]
        transcript = [row["transcript"] for row in batch]
        file_name = [row["file_name"] for row in batch]

        return {'audio': audio, 'transcript': transcript, 'file_name': file_name}

    return DataLoader(dataset, collate_fn=collate_fn, batch_size=batch_size)


# -----------------------------------------------------------------------------------------------
def extract_features_and_labels(dataset, processor, batch_size=32, remove_other_columns=False) -> Dataset:

    def prepare_func(batch):
        features = processor.feature_extractor(batch["audio"], sampling_rate=16000, return_attention_mask=True)
        batch["input_features"] = features.input_features
        batch["attention_mask"] = features.attention_mask
        batch["labels"] = processor.tokenizer(batch["transcript"]).input_ids
        return batch

    if remove_other_columns:
        dataset = dataset.map(prepare_func, batched=True, batch_size=batch_size, remove_columns=dataset.column_names)
    else:
        dataset = dataset.map(prepare_func, batched=True, batch_size=batch_size)
    return dataset
