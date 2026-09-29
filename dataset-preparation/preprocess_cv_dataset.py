import argparse
import os
import regex
import unicodedata
import numpy as np
import pandas as pd
import torchaudio
from scipy.io.wavfile import write
from tqdm import tqdm
import csv
import sys
from omegaconf import OmegaConf


# ----------------------------------------------------------------------------------
def load_tsv_files(input_dir: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load train, test, and dev TSV files from Common Voice dataset."""
    train_df = pd.read_csv(os.path.join(input_dir, "train.tsv"), sep="\t", engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    test_df = pd.read_csv(os.path.join(input_dir, "test.tsv"), sep="\t", engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    dev_df = pd.read_csv(os.path.join(input_dir, "dev.tsv"), sep="\t", engine="python", quoting=csv.QUOTE_NONE, on_bad_lines="skip")
    
    print(f"Loaded datasets:")
    print(f"  Train: {len(train_df)} samples")
    print(f"  Test: {len(test_df)} samples")
    print(f"  Dev: {len(dev_df)} samples")
    
    return train_df, test_df, dev_df


# ----------------------------------------------------------------------------------
def load_durations(input_dir: str) -> pd.DataFrame:
    """Load clip durations from Common Voice dataset."""
    durations_path = os.path.join(input_dir, "clip_durations.tsv")
    
    if not os.path.exists(durations_path):
        print("Warning: clip_durations.tsv not found. Duration filtering will be skipped.")
        return None
    
    durations_df = pd.read_csv(durations_path, sep="\t")
    durations_df = durations_df.rename(columns={"clip": "file_name", "duration[ms]": "duration"})
    durations_df["duration"] = durations_df["duration"] / 1000  # Convert to seconds
    
    return durations_df


# ----------------------------------------------------------------------------------
def prepare_dataframe(df: pd.DataFrame, durations_df: pd.DataFrame, max_duration: float) -> pd.DataFrame:
    """
    Prepare dataframe: select columns, rename, filter by duration.
    
    Args:
        df: Input dataframe from TSV
        durations_df: Dataframe with clip durations
        max_duration: Maximum duration in seconds
    
    Returns:
        Prepared dataframe
    """
    # Select and rename columns
    df = df[["path", "sentence"]].copy()
    df = df.rename(columns={"path": "file_name", "sentence": "transcript"})
    
    # Drop NA values
    df = df.dropna()
    
    # Filter by duration if durations are available
    if durations_df is not None:
        df = pd.merge(df, durations_df, on="file_name", how="inner")
        df = df[df["duration"] < max_duration]
    
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------------
def normalize_transcript(text: str) -> str:
    """
    Normalize transcript for ASR training/evaluation.
    
    Normalization steps:
    - Unicode NFC normalization
    - Convert to lowercase
    - Remove punctuation
    - Normalize whitespace
    
    Args:
        text: Input transcript string
    
    Returns:
        Normalized transcript string
    """
    if pd.isna(text):
        return ""

    # Unicode normalization
    text = unicodedata.normalize("NFC", text)

    # Convert to lowercase
    text = text.lower()

    # Remove Unicode punctuation only
    text = regex.sub(r"\p{P}+", "", text)

    # Normalize whitespace (preserve ZWNJ)
    text = regex.sub(r"[ \t\r\n]+", " ", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ----------------------------------------------------------------------------------
def check_unusual_samples(df: pd.DataFrame) -> pd.Series:
    """
    Check for unusual samples in a dataframe.
    
    Args:
        df: DataFrame with 'transcript' column
    
    Returns:
        Boolean mask for unusual rows
    """
    # Check for tabs
    tabs_mask = df['transcript'].str.contains(r'\t', regex=True, na=False)
    # Check for digits
    digits_mask = df['transcript'].str.contains(r'[0-9\u0660-\u0669\u06F0-\u06F9]', regex=True, na=False)
    
    unusual_mask = tabs_mask | digits_mask
    
    return unusual_mask


# ----------------------------------------------------------------------------------
def process_audio_files(
    df: pd.DataFrame,
    split_name: str,
    input_clips_dir: str,
    output_dir: str,
    target_sample_rate: int = 16000
):
    """
    Process audio files: resample to target sample rate and save as WAV.
    
    Args:
        df: DataFrame with 'file_name' column
        split_name: 'train', 'test', or 'dev'
        input_clips_dir: Directory containing original audio clips
        output_dir: Output directory for processed dataset
        target_sample_rate: Target sample rate (default: 16000)
    
    Returns:
        None
    """
    print(f"\nProcessing {split_name} audio files...")
    
    output_audio_dir = os.path.join(output_dir, split_name, "audio")
    os.makedirs(output_audio_dir, exist_ok=True)
    
    # Create resampler for common sample rate (48kHz is typical for Common Voice)
    resampler_48k = torchaudio.transforms.Resample(orig_freq=48000, new_freq=target_sample_rate)
    
    for file_name in tqdm(df['file_name'], desc=f"Processing {split_name}"):
        try:
            input_path = os.path.join(input_clips_dir, file_name)
            
            # Load audio
            waveform, sample_rate = torchaudio.load(input_path)
            
            # Resample if needed
            if sample_rate != target_sample_rate:
                if sample_rate == 48000:
                    waveform = resampler_48k(waveform)
                else:
                    resampler_custom = torchaudio.transforms.Resample(
                        orig_freq=sample_rate, new_freq=target_sample_rate
                    )
                    waveform = resampler_custom(waveform)
            
            # Convert to mono if stereo
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)
            
            # Convert to numpy and scale to int16
            audio_np = waveform.squeeze().numpy()
            audio_np = (audio_np * 32767).astype(np.int16)
            
            # Save as WAV file
            output_filename = file_name.replace('.mp3', '.wav')
            output_path = os.path.join(output_audio_dir, output_filename)
            write(output_path, target_sample_rate, audio_np)
            
        except Exception as error:
            print(f"  {file_name}: {error}")
            sys.exit(1)


# ----------------------------------------------------------------------------------
def process_split(
    df: pd.DataFrame,
    split_name: str,
    durations_df: pd.DataFrame,
    input_clips_dir: str,
    output_dir: str,
    max_duration: float,
    num_samples: int,
    seed: int,
    target_sample_rate: int = 16000
) -> pd.DataFrame:
    """
    Process a single split (train/test/dev).
    
    Args:
        df: Input dataframe from TSV
        split_name: 'train', 'test', or 'dev'
        durations_df: Dataframe with clip durations
        input_clips_dir: Directory containing original audio clips
        output_dir: Output directory
        max_duration: Maximum duration in seconds
        num_samples: Number of samples to store
        target_sample_rate: Target sample rate for audio
    
    Returns:
        Processed and cleaned DataFrame
    """
    print(f"\n{'='*60}")
    print(f"Processing {split_name.upper()} split")
    print(f"{'='*60}")
    
    # Prepare dataframe
    df = prepare_dataframe(df, durations_df, max_duration)
    print(f"After duration filtering: {len(df)} samples")
    
    # Check for unusual samples
    unusual_mask = check_unusual_samples(df)
    unusual_count = unusual_mask.sum()
    print(f"Unusual samples found: {unusual_count}")
    
    # Remove unusual samples
    df = df[~unusual_mask].reset_index(drop=True)
    
    # Normalize transcriptions
    print("Normalizing transcriptions...")
    df['transcript'] = df['transcript'].apply(normalize_transcript)
    
    # Remove empty transcripts
    empty_count = (df['transcript'].str.len() == 0).sum()
    if empty_count > 0:
        print(f"Removing {empty_count} empty transcripts after normalization")
        df = df[df['transcript'].str.len() > 0].reset_index(drop=True)
    
    # Keep only file_name and transcript columns
    df = df[['file_name', 'transcript']]

    # Sample df. If num_samples is omitted, keep all filtered samples in a deterministic order.
    if num_samples is None:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        num_samples = min(int(num_samples), len(df))
        df = df.sample(n=num_samples, random_state=seed).reset_index(drop=True)

    # Process audio
    process_audio_files(df, split_name, input_clips_dir, output_dir, target_sample_rate)

    # Update file extensions from .mp3 to .wav
    df['file_name'] = df['file_name'].str.replace('.mp3', '.wav', regex=False)

    # Save transcript
    transcript_path = os.path.join(output_dir, split_name, "transcript.csv")
    os.makedirs(os.path.dirname(transcript_path), exist_ok=True)
    df.to_csv(transcript_path, index=False, encoding="utf-8")
    print(f"Saved transcript: {transcript_path}")
    print(f"Final sample count: {len(df)}")
    
    return df


# ----------------------------------------------------------------------------------
def verify_dataset(output_dir: str):
    """Verify the processed dataset."""
    print(f"\n{'='*60}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*60}")
    
    for split_name in ['train', 'test', 'dev']:
        transcript_path = os.path.join(output_dir, split_name, "transcript.csv")
        
        if not os.path.exists(transcript_path):
            print(f"\n{split_name}: Transcript file not found")
            continue
        
        df = pd.read_csv(transcript_path)
        
        # Check for issues
        has_tabs = df['transcript'].str.contains(r'\t', regex=True, na=False).sum()
        has_digits = df['transcript'].str.contains(r'[0-9\u0660-\u0669\u06F0-\u06F9]', regex=True, na=False).sum()
        has_punctuation = df['transcript'].apply(lambda x: bool(regex.search(r"\p{P}+", x))).sum()
        has_uppercase = df['transcript'].str.contains(r'[A-Z]', regex=True, na=False).sum()
        has_empty = (df['transcript'].str.len() == 0).sum()
        
        print(f"\n{split_name.upper()} ({len(df)} samples):")
        print(f"  ✓ Tabs: {has_tabs}")
        print(f"  ✓ Digits: {has_digits}")
        print(f"  ✓ Punctuation: {has_punctuation}")
        print(f"  ✓ Uppercase: {has_uppercase}")
        print(f"  ✓ Empty transcripts: {has_empty}")
        
        if len(df) > 0:
            print(f"  Sample: '{df['transcript'].iloc[0]}'")


# ----------------------------------------------------------------------------------

# Load the configs
parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()
cfg = OmegaConf.load(args.config)

# Validate input directory
if not os.path.exists(cfg.input_dir):
    print(f"Error: Input directory not found: {cfg.input_dir}")
    sys.exit(1)

# Create output directory
os.makedirs(cfg.output_dir, exist_ok=True)

# Load TSV files
train_df, test_df, dev_df = load_tsv_files(cfg.input_dir)

# Load durations
durations_df = load_durations(cfg.input_dir)

# Input clips directory
input_clips_dir = os.path.join(cfg.input_dir, "clips")

if not os.path.exists(input_clips_dir):
    print(f"Error: Clips directory not found: {input_clips_dir}")
    sys.exit(1)

# Process each split
split_dfs = {"train": train_df, "test": test_df, "dev": dev_df}

num_samples = {"train": cfg.train_num_samples, "test": cfg.test_num_samples, "dev": cfg.dev_num_samples}

for split_name in ["train", "test", "dev"]:
    process_split(
        df=split_dfs[split_name],
        split_name=split_name,
        durations_df=durations_df,
        input_clips_dir=input_clips_dir,
        output_dir=cfg.output_dir,
        max_duration=cfg.max_duration,
        num_samples=num_samples[split_name],
        seed=cfg.seed,
        target_sample_rate=cfg.sample_rate
    )

# Verify results
verify_dataset(cfg.output_dir)

print(f"\n{'=' * 60}")
print("Processing complete!")
print(f"{'=' * 60}")
