"""
loader.py — Chargement et nettoyage structurel des données PubHealth.
Source : Hugging Face bigbio/pubhealth, config pubhealth_bigbio_pairs, via Parquet direct.
Responsabilite : I/O uniquement.
"""
import pandas as pd
from datasets import load_dataset

VALID_LABELS = ['true', 'false', 'mixture', 'unproven']

PARQUET_BASE = "https://huggingface.co/datasets/bigbio/pubhealth/resolve/refs%2Fconvert%2Fparquet/pubhealth_bigbio_pairs"
DATA_FILES = {
    "train": f"{PARQUET_BASE}/train/0000.parquet",
    "validation": f"{PARQUET_BASE}/validation/0000.parquet",
    "test": f"{PARQUET_BASE}/test/0000.parquet",
}


def load_raw_data():
    ds = load_dataset("parquet", data_files=DATA_FILES)
    df_train = ds['train'].to_pandas()
    df_val = ds['validation'].to_pandas()
    df_test = ds['test'].to_pandas()
    return (df_train.reset_index(drop=True),
            df_val.reset_index(drop=True),
            df_test.reset_index(drop=True))


def clean_corrupted_rows(df):
    return df[df['label'].isin(VALID_LABELS)].reset_index(drop=True)


def clean_claim_text(text):
    text = str(text).strip()
    text = text.strip('"')
    while '"""' in text:
        text = text.replace('"""', '"')
    return text.strip()


def rename_columns(df):
    df = df.copy()
    df = df.rename(columns={'text_1': 'claim'})
    df['claim'] = df['claim'].apply(clean_claim_text)
    return df[['claim', 'label']]


def load_and_clean_data():
    df_train, df_val, df_test = load_raw_data()
    df_train = rename_columns(clean_corrupted_rows(df_train))
    df_val = rename_columns(clean_corrupted_rows(df_val))
    df_test = rename_columns(clean_corrupted_rows(df_test))
    return df_train, df_val, df_test