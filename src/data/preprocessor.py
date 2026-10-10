"""
preprocessor.py — Transformation ML : tokenization, encodage labels, poids de classe.
Responsabilité : tout ce qui connaît le tokenizer/modèle (ML-aware).
"""
import torch
from transformers import AutoTokenizer

MAX_LENGTH = 64  # validé en exploration : 99e percentile ≈ 40 tokens, 0,51% de troncature
VALID_LABELS = ['true', 'false', 'mixture', 'unproven']
LABEL2ID = {label: idx for idx, label in enumerate(VALID_LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def encode_labels(df):
    """Transforme les labels textuels en identifiants numériques (0-3), attendus par PyTorch."""
    df = df.copy()
    df['label_id'] = df['label'].map(LABEL2ID)
    return df


def tokenize_data(df, tokenizer_name="bert-base-uncased"):
    """
    Tokenize les claims avec le tokenizer du modèle choisi (BERT ou BioBERT).
    - max_length=64        : longueur fixe validée en exploration
    - padding='max_length' : uniformise les séquences courtes (ajout de [PAD])
    - truncation=True      : coupe les rares séquences dépassant 64 tokens
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    encodings = tokenizer(
        df['claim'].tolist(),
        max_length=MAX_LENGTH,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encodings, tokenizer


def compute_class_weights(df):
    """Calcule les poids de classe (inversement proportionnels) pour compenser le déséquilibre."""
    class_counts = df['label'].value_counts()
    total = len(df)
    n_classes = len(class_counts)
    weights = {label: total / (n_classes * count) for label, count in class_counts.items()}
    return torch.tensor([weights[ID2LABEL[i]] for i in range(n_classes)])


class PubHealthDataset(torch.utils.data.Dataset):
    """Wrapper PyTorch Dataset — assemble input_ids, attention_mask et labels pour le DataLoader."""
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


def prepare_data(df_train, df_val, df_test, tokenizer_name="bert-base-uncased"):
    """
    Point d'entrée unique : prend des DataFrames déjà chargés/nettoyés (via loader.py)
    et retourne des Dataset PyTorch prêts pour l'entraînement, ainsi que
    les poids de classe et le tokenizer utilisé.
    """
    df_train = encode_labels(df_train)
    df_val = encode_labels(df_val)
    df_test = encode_labels(df_test)

    train_encodings, tokenizer = tokenize_data(df_train, tokenizer_name)
    val_encodings, _ = tokenize_data(df_val, tokenizer_name)
    test_encodings, _ = tokenize_data(df_test, tokenizer_name)

    train_dataset = PubHealthDataset(train_encodings, df_train['label_id'].tolist())
    val_dataset = PubHealthDataset(val_encodings, df_val['label_id'].tolist())
    test_dataset = PubHealthDataset(test_encodings, df_test['label_id'].tolist())

    class_weights = compute_class_weights(df_train)

    return train_dataset, val_dataset, test_dataset, class_weights, tokenizer