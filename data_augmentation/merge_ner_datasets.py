"""
Script to merge the original HuggingFace NER dataset with augmented data.
Creates a combined dataset for fine-tuning BERT-based NER models.
"""

import os
import pandas as pd
import json
from huggingface_hub import login

def load_huggingface_dataset(num_samples: int = 2500) -> pd.DataFrame:
    """Load a subset of the original HuggingFace NER dataset."""
    print(f"Loading {num_samples} samples from HuggingFace dataset...")
    
    # Login to Hugging Face
    hf_token = os.getenv("HUGGINFACE")
    if hf_token:
        login(token=hf_token)
    
    # Load the dataset
    splits = {'train': 'train.jsonl', 'eval': 'eval.jsonl'}
    df = pd.read_json("hf://datasets/empathyai/books-ner-dataset/" + splits["train"], lines=True)
    
    # Sample a subset
    df_sampled = df.sample(n=min(num_samples, len(df)), random_state=42)
    
    print(f"Loaded {len(df_sampled)} samples from HuggingFace")
    return df_sampled


def load_augmented_dataset(path: str = "./data/ner_data_augmentation.csv") -> pd.DataFrame:
    """Load the augmented NER dataset."""
    print(f"Loading augmented dataset from {path}...")
    
    df = pd.read_csv(path)
    
    # Parse JSON strings back to lists
    df['tokenized_text'] = df['tokenized_text'].apply(json.loads)
    df['ner'] = df['ner'].apply(json.loads)
    
    print(f"Loaded {len(df)} augmented samples")
    return df


def merge_datasets(df_original: pd.DataFrame, df_augmented: pd.DataFrame) -> pd.DataFrame:
    """Merge original and augmented datasets."""
    print("Merging datasets...")
    
    # Ensure both have the same columns
    # Original has: tokenized_text, ner
    # Augmented has: tokenized_text, ner
    
    df_merged = pd.concat([df_original, df_augmented], ignore_index=True)
    
    # Shuffle the merged dataset
    df_merged = df_merged.sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"Merged dataset size: {len(df_merged)}")
    return df_merged


def create_train_eval_split(df: pd.DataFrame, eval_ratio: float = 0.1) -> tuple:
    """Split the dataset into train and eval sets."""
    print(f"Splitting dataset (eval_ratio={eval_ratio})...")
    
    eval_size = int(len(df) * eval_ratio)
    
    df_eval = df.iloc[:eval_size]
    df_train = df.iloc[eval_size:]
    
    print(f"Train size: {len(df_train)}, Eval size: {len(df_eval)}")
    return df_train, df_eval


def save_datasets(df_train: pd.DataFrame, df_eval: pd.DataFrame, output_dir: str = "./data"):
    """Save train and eval datasets as JSONL files (HuggingFace format)."""
    
    train_path = os.path.join(output_dir, "ner_train_merged.jsonl")
    eval_path = os.path.join(output_dir, "ner_eval_merged.jsonl")
    
    # Save as JSONL (same format as original HuggingFace dataset)
    df_train.to_json(train_path, orient='records', lines=True)
    df_eval.to_json(eval_path, orient='records', lines=True)
    
    print(f"Saved train dataset to: {train_path}")
    print(f"Saved eval dataset to: {eval_path}")
    
    # Also save combined CSV for convenience
    csv_path = os.path.join(output_dir, "ner_combined.csv")
    df_combined = pd.concat([df_train, df_eval], ignore_index=True)
    df_combined['tokenized_text'] = df_combined['tokenized_text'].apply(json.dumps)
    df_combined['ner'] = df_combined['ner'].apply(json.dumps)
    df_combined.to_csv(csv_path, index=False)
    print(f"Saved combined CSV to: {csv_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Merge NER datasets for fine-tuning")
    parser.add_argument("--hf_samples", type=int, default=2500, 
                        help="Number of samples from HuggingFace dataset (default: 2500)")
    parser.add_argument("--augmented_path", type=str, default="./data/ner_data_augmentation.csv",
                        help="Path to augmented dataset")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="Output directory for merged datasets")
    parser.add_argument("--eval_ratio", type=float, default=0.1,
                        help="Ratio of data to use for evaluation (default: 0.1)")
    args = parser.parse_args()
    
    # Load datasets
    df_original = load_huggingface_dataset(args.hf_samples)
    df_augmented = load_augmented_dataset(args.augmented_path)
    
    # Merge
    df_merged = merge_datasets(df_original, df_augmented)
    
    # Split into train/eval
    df_train, df_eval = create_train_eval_split(df_merged, args.eval_ratio)
    
    # Save
    save_datasets(df_train, df_eval, args.output_dir)
    
    # Print summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    print(f"Original HuggingFace samples: {len(df_original)}")
    print(f"Augmented samples:            {len(df_augmented)}")
    print(f"Total merged:                 {len(df_merged)}")
    print(f"Train set:                    {len(df_train)}")
    print(f"Eval set:                     {len(df_eval)}")
    print("="*50)


if __name__ == "__main__":
    main()
