import argparse
import os
import regex
import sys
import unicodedata
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torchaudio
from omegaconf import OmegaConf
from scipy.io.wavfile import write
from tqdm import tqdm


SUPPORTED_AUDIO_EXTENSIONS = {".flac", ".opus", ".wav", ".mp3"}


# ----------------------------------------------------------------------------------
def normalize_transcript(text: str) -> str:
    """Apply the transcript normalization used by this repository."""
    if pd.isna(text):
        return ""

    text = unicodedata.normalize("NFC", str(text))
    text = text.lower()
    text = regex.sub(r"\p{P}+", "", text)
    text = regex.sub(r"[ \t\r\n]+", " ", text)
    return text.strip()


# ----------------------------------------------------------------------------------
def has_unusual_content(text: str) -> bool:
    """Match the Common Voice preprocessing filter for tabs and digits."""
    if pd.isna(text):
        return True
    text = str(text)
    has_tab = bool(regex.search(r"\t", text))
    has_digit = bool(regex.search(r"[0-9\u0660-\u0669\u06F0-\u06F9]", text))
    return has_tab or has_digit


# ----------------------------------------------------------------------------------
def load_transcripts(split_dir: Path) -> pd.DataFrame:
    """Load an MLS ``transcripts.txt`` file.

    MLS transcript lines are expected to contain an utterance ID followed by a
    transcript, normally separated by a tab. A whitespace fallback is accepted
    to make the parser tolerant to equivalent local exports.
    """
    transcript_path = split_dir / "transcripts.txt"
    if not transcript_path.is_file():
        raise FileNotFoundError(f"Missing MLS transcript file: {transcript_path}")

    records = []
    with transcript_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.rstrip("\n\r")
            if not line:
                continue

            if "\t" in line:
                utterance_id, transcript = line.split("\t", 1)
            else:
                parts = line.split(maxsplit=1)
                if len(parts) != 2:
                    raise ValueError(
                        f"Could not parse {transcript_path}:{line_number}: {line!r}"
                    )
                utterance_id, transcript = parts

            records.append({
                "utterance_id": utterance_id.strip(),
                "transcript": transcript.strip(),
            })

    if not records:
        raise ValueError(f"No transcripts found in {transcript_path}")

    return pd.DataFrame(records)


# ----------------------------------------------------------------------------------
def index_audio_files(split_dir: Path) -> Dict[str, Path]:
    """Index MLS audio recursively by utterance ID (file stem)."""
    audio_root = split_dir / "audio"
    if not audio_root.is_dir():
        raise FileNotFoundError(f"Missing MLS audio directory: {audio_root}")

    index = {}
    for path in audio_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_AUDIO_EXTENSIONS:
            continue
        utterance_id = path.stem
        if utterance_id in index:
            raise ValueError(
                f"Duplicate audio utterance ID {utterance_id!r} in {audio_root}"
            )
        index[utterance_id] = path

    if not index:
        raise ValueError(f"No supported audio files found under {audio_root}")

    return index


# ----------------------------------------------------------------------------------
def get_duration_seconds(audio_path: Path) -> float:
    """Read audio duration without decoding the full waveform when possible."""
    try:
        info = torchaudio.info(str(audio_path))
        if info.sample_rate > 0 and info.num_frames > 0:
            return info.num_frames / info.sample_rate
    except Exception:
        pass

    waveform, sample_rate = torchaudio.load(str(audio_path))
    return waveform.shape[-1] / sample_rate


# ----------------------------------------------------------------------------------
def prepare_split_dataframe(
    split_dir: Path,
    max_duration: float,
    num_samples: Optional[int],
    seed: int,
) -> pd.DataFrame:
    transcripts = load_transcripts(split_dir)
    audio_index = index_audio_files(split_dir)

    transcripts["audio_path"] = transcripts["utterance_id"].map(audio_index)
    missing = transcripts["audio_path"].isna().sum()
    if missing:
        print(f"Warning: skipping {missing} transcript(s) without matching audio files.")
        transcripts = transcripts.dropna(subset=["audio_path"]).reset_index(drop=True)

    unusual_mask = transcripts["transcript"].apply(has_unusual_content)
    if unusual_mask.any():
        print(f"Skipping {int(unusual_mask.sum())} transcript(s) containing tabs or digits.")
        transcripts = transcripts[~unusual_mask].reset_index(drop=True)

    transcripts["transcript"] = transcripts["transcript"].apply(normalize_transcript)
    transcripts = transcripts[transcripts["transcript"].str.len() > 0].reset_index(drop=True)

    if max_duration is not None:
        keep = []
        print(f"Checking durations (< {max_duration} s)...")
        for path in tqdm(transcripts["audio_path"], desc="Duration filter"):
            keep.append(get_duration_seconds(Path(path)) < max_duration)
        transcripts = transcripts[np.asarray(keep, dtype=bool)].reset_index(drop=True)

    if num_samples is None:
        transcripts = transcripts.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    else:
        n = min(int(num_samples), len(transcripts))
        transcripts = transcripts.sample(n=n, random_state=seed).reset_index(drop=True)

    return transcripts


# ----------------------------------------------------------------------------------
def convert_audio(
    df: pd.DataFrame,
    output_audio_dir: Path,
    target_sample_rate: int,
) -> list[str]:
    output_audio_dir.mkdir(parents=True, exist_ok=True)
    output_names = []

    for row in tqdm(df.itertuples(index=False), total=len(df), desc="Converting audio"):
        input_path = Path(row.audio_path)
        try:
            waveform, sample_rate = torchaudio.load(str(input_path))

            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)

            if sample_rate != target_sample_rate:
                waveform = torchaudio.functional.resample(
                    waveform, orig_freq=sample_rate, new_freq=target_sample_rate
                )

            waveform = waveform.squeeze(0).clamp(-1.0, 1.0)
            audio_np = (waveform.cpu().numpy() * 32767).astype(np.int16)

            output_name = f"{row.utterance_id}.wav"
            write(output_audio_dir / output_name, target_sample_rate, audio_np)
            output_names.append(output_name)
        except Exception as error:
            print(f"Failed to process {input_path}: {error}")
            sys.exit(1)

    return output_names


# ----------------------------------------------------------------------------------
def process_split(
    input_dir: Path,
    output_dir: Path,
    split_name: str,
    max_duration: float,
    num_samples: Optional[int],
    seed: int,
    target_sample_rate: int,
):
    print(f"\n{'=' * 60}")
    print(f"Processing MLS {split_name.upper()} split")
    print(f"{'=' * 60}")

    split_dir = input_dir / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing MLS split directory: {split_dir}")

    df = prepare_split_dataframe(
        split_dir=split_dir,
        max_duration=max_duration,
        num_samples=num_samples,
        seed=seed,
    )

    output_split_dir = output_dir / split_name
    output_names = convert_audio(
        df=df,
        output_audio_dir=output_split_dir / "audio",
        target_sample_rate=target_sample_rate,
    )

    output_df = pd.DataFrame({
        "file_name": output_names,
        "transcript": df["transcript"].tolist(),
    })
    output_split_dir.mkdir(parents=True, exist_ok=True)
    transcript_path = output_split_dir / "transcript.csv"
    output_df.to_csv(transcript_path, index=False, encoding="utf-8")

    print(f"Saved transcript: {transcript_path}")
    print(f"Final sample count: {len(output_df)}")


# ----------------------------------------------------------------------------------
def verify_dataset(output_dir: Path):
    print(f"\n{'=' * 60}")
    print("VERIFICATION SUMMARY")
    print(f"{'=' * 60}")

    for split_name in ("train", "dev", "test"):
        transcript_path = output_dir / split_name / "transcript.csv"
        audio_dir = output_dir / split_name / "audio"
        if not transcript_path.is_file():
            print(f"{split_name}: missing transcript.csv")
            continue

        df = pd.read_csv(transcript_path)
        missing_audio = sum(not (audio_dir / name).is_file() for name in df["file_name"])
        print(
            f"{split_name}: {len(df)} samples, "
            f"{missing_audio} missing audio file(s)"
        )


# ----------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)

    if not cfg.input_dir:
        raise ValueError("input_dir must not be empty")
    if not cfg.output_dir:
        raise ValueError("output_dir must not be empty")

    input_dir = Path(cfg.input_dir).expanduser()
    output_dir = Path(cfg.output_dir).expanduser()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"MLS input_dir does not exist: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    split_sample_counts = {
        "train": cfg.get("train_num_samples", None),
        "dev": cfg.get("dev_num_samples", None),
        "test": cfg.get("test_num_samples", None),
    }

    for split_name in ("train", "dev", "test"):
        process_split(
            input_dir=input_dir,
            output_dir=output_dir,
            split_name=split_name,
            max_duration=cfg.get("max_duration", None),
            num_samples=split_sample_counts[split_name],
            seed=int(cfg.seed),
            target_sample_rate=int(cfg.sample_rate),
        )

    verify_dataset(output_dir)


if __name__ == "__main__":
    main()
